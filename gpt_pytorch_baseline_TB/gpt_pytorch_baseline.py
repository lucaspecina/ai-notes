"""
Refactor de gpt_pytorch_baseline.py para poder importarlo desde una notebook.

Uso rápido desde Jupyter/Colab:

    from gpt_pytorch_baseline import main, build_run, train_run, run_profiler_window

    # 1) correr todo con defaults
    model, tokenizer, metrics, run = main(
        data_path="data/input.txt",
        num_epochs=1, eval_freq=100, eval_iter=10, max_steps=1000,
    )

    # 2) o bien en 2 pasos (con más control)
    run = build_run(
        data_path="data/input.txt",
        gpt_config={"context_length":256, "emb_dim":512, "n_heads":8, "n_layers":12, "drop_rate":0.1, "qkv_bias":False},
        settings={"learning_rate":5e-4, "batch_size":16, "weight_decay":0.1, "seed":20},
        perf_cfg=None,                 # usa defaults de PerfConfig
        run_name=None,                 # para nombrar el run_dir
        csv_path="./perf_logs/train_metrics.csv",
        enable_sysmon=True,
    )
    metrics = train_run(run, num_epochs=1, eval_freq=100, eval_iter=10, max_steps=1000, save_chkpt=True)

TensorBoard:
    tensorboard --logdir ./tb_traces

Este archivo NO ejecuta nada al importarse. Solo define clases y funciones.
"""

from __future__ import annotations
import os, time, math, platform, contextlib, ctypes
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from IPython.display import display, Markdown

# ---- perfkit: monitores / perfiles (archivo separado perfkit.py en el mismo directorio) ----
try:
    from perfkit import PerfMonitor, PerfConfig, GPUSystemMonitor
except Exception:  # permite importar aunque no esté perfkit
    PerfMonitor = PerfConfig = GPUSystemMonitor = None


# ============================================================
# Utils básicos
# ============================================================

def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Datos
# ============================================================

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids)


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def load_text_and_tokenize(data_path: str) -> Tuple[str, CharTokenizer, torch.Tensor]:
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return text, tokenizer, data


def make_loaders(train_data: torch.Tensor, val_data: torch.Tensor, block_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    use_cuda = torch.cuda.is_available()
    on_windows = (platform.system() == "Windows")

    train_dataset = CharDataset(train_data, block_size)
    val_dataset   = CharDataset(val_data,   block_size)

    def _mk(dataset, shuffle):
        if on_windows:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=use_cuda)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=(4 if use_cuda else 0), pin_memory=use_cuda,
                              prefetch_factor=(2 if use_cuda else None),
                              persistent_workers=(True if use_cuda else False))

    return _mk(train_dataset, True), _mk(val_dataset, False)


def get_calibration_batch(dataset: Dataset, batch_size: int) -> torch.Tensor:
    """Devuelve 1 batch con num_workers=0 (robusto en Windows). No lo mueve a device."""
    cal_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    x, _ = next(iter(cal_loader))
    return x


# ============================================================
# Modelo
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # proyecciones QKV
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # split en cabezas
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # atención causal
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # salida
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        emb = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.GELU(),
            nn.Linear(4 * emb, emb),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Atención + residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # FFN + residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = dict(cfg)  # guardar config dentro del modelo (para checkpoints)
        self.device_hint = cfg.get("device", pick_device())

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])] )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx: (B, T)
        B, T = in_idx.shape
        tok = self.tok_emb(in_idx)
        pos = self.pos_emb(torch.arange(T, device=in_idx.device))
        x = self.drop_emb(tok + pos)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits  # **IMPORTANTE**: devolver logits


# ============================================================
# Checkpoints y generación
# ============================================================

def get_checkpoint_filename(cfg: Dict[str, Any]) -> str:
    return f"model_checkpoints/gpt_checkpoint_ctx{cfg['context_length']}_emb{cfg['emb_dim']}_heads{cfg['n_heads']}_layers{cfg['n_layers']}_tokchar.pth"


def save_checkpoint(model: GPTModel, optimizer: torch.optim.Optimizer, epoch: int, filename: str):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'gpt_config': model.cfg,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model: GPTModel, optimizer: torch.optim.Optimizer, filename: str) -> int:
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=model.device_hint)
        ck_cfg = checkpoint.get('gpt_config', {})
        if ck_cfg and ck_cfg != model.cfg:
            print(f"[warn] Config mismatch between model and checkpoint {filename}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = int(checkpoint['epoch'])
        print(f"Checkpoint loaded from {filename}, starting from epoch {epoch + 1}")
        return epoch
    else:
        print(f"No checkpoint found at {filename}, starting from scratch")
        return -1


def load_model_for_inference(gpt_config: Dict[str, Any], filename: str, device: torch.device) -> GPTModel:
    model = GPTModel(gpt_config).to(device)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        ck_cfg = checkpoint.get('gpt_config', {})
        if ck_cfg and ck_cfg != gpt_config:
            print(f"[warn] Config mismatch in checkpoint {filename}")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filename} for inference")
    else:
        print(f"No checkpoint found at {filename}; returning untrained model")
    return model


def generate_text_simple(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=200):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    # Use tokenizer's encode method
    encoded = torch.tensor([tokenizer.encode(start_context)], dtype=torch.long, device=device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=max_new_tokens, context_size=context_size
        )
        # Use tokenizer's decode method
        decoded_text = tokenizer.decode(token_ids[0].tolist())
        # decoded_text = decoded_text.replace("\n", " ")  # Compact print format
    model.train()
    display(Markdown(f"**Generated Output:**\n\n{decoded_text}"))
    return decoded_text


# ============================================================
# Pérdida y evaluación
# ============================================================

def calc_loss_batch(input_batch, target_batch, model: GPTModel, device: torch.device):
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader: DataLoader, model: GPTModel, device: torch.device, num_batches: Optional[int] = None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    if model_was_training:
        model.train()
    return total_loss / num_batches


def evaluate_model(model: GPTModel, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, eval_iter: int):
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    if model_was_training:
        model.train()
    return train_loss, val_loss


# ============================================================
# Profiler (opcional, exporta a TensorBoard)
# ============================================================

def run_profiler_window(model: GPTModel, loader: DataLoader, device: torch.device, steps: int = 10, tb_logdir: str = "./tb_traces/profile"):
    from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
    model.train()
    sched = schedule(wait=1, warmup=2, active=max(1, steps - 3), repeat=1)

    it = iter(loader)
    def _one_step():
        inp, tgt = next(it)
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        loss = calc_loss_batch(inp, tgt, model, device)
        loss.backward()
        model.zero_grad(set_to_none=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(tb_logdir),
    ) as prof:
        for _ in range(steps):
            _one_step()
            prof.step()
    print(f"[profiler] Trace guardado en {tb_logdir}")


# ============================================================
# Entrenamiento (API estable)
# ============================================================

class TrainMetrics(NamedTuple):
    train_losses: List[float]
    val_losses: List[float]
    tokens_seen: List[int]


def train(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    max_steps: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    save_chkpt: bool = False,
    monitor: Optional[PerfMonitor] = None,
) -> TrainMetrics:
    """Entrenamiento single-GPU con instrumentación opcional (PerfMonitor)."""

    @contextlib.contextmanager
    def _null_cm():
        yield

    class _NullStepCtx:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
        def phase(self, name): return _null_cm()

    def _step_ctx(tokens_in_step):
        if monitor is None:
            return _NullStepCtx()
        return monitor.step(tokens_in_step)

    train_losses: List[float] = []
    val_losses: List[float] = []
    track_tokens: List[int] = []
    global_step = -1
    stop_training = False

    # cargar checkpoint si corresponde
    start_epoch = 0
    if checkpoint_path:
        try:
            last_epoch = load_checkpoint(model, optimizer, checkpoint_path)
            if isinstance(last_epoch, int) and last_epoch >= 0:
                start_epoch = last_epoch + 1
        except Exception as e:
            print(f"[warn] no se pudo cargar checkpoint ({e}). iniciando desde 0.")

    model.train()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        for inp_batch, tgt_batch in train_loader:
            global_step += 1
            B, T = inp_batch.shape
            tokens_in_step = B * T

            with _step_ctx(tokens_in_step) as s:
                # DATA
                with s.phase("data"):
                    inp_batch = inp_batch.to(device, non_blocking=True)
                    tgt_batch = tgt_batch.to(device, non_blocking=True)
                # FORWARD
                with s.phase("forward"):
                    loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
                # BACKWARD
                with s.phase("backward"):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                # OPT
                with s.phase("optim"):
                    optimizer.step()

            # evaluación periódica
            if eval_freq and (global_step % eval_freq == 0):
                tr, va = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(tr)
                val_losses.append(va)
                track_tokens.append(tokens_in_step if not track_tokens else track_tokens[-1] + tokens_in_step)
                if monitor is not None:
                    monitor.log_eval(tr, va, optimizer)
                else:
                    lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")
                    print(f"[eval] ep={epoch+1} step={global_step:06d} train={tr:.3f} val={va:.3f} lr={lr:.2e}")

            if (max_steps is not None) and (global_step >= max_steps):
                print(f"Reached max steps: {max_steps}")
                stop_training = True
                break

        if save_chkpt and checkpoint_path:
            try:
                save_checkpoint(model, optimizer, epoch, checkpoint_path)
            except Exception as e:
                print(f"[warn] no se pudo guardar checkpoint ({e})")
        if stop_training:
            break

    return TrainMetrics(train_losses, val_losses, track_tokens)


# ============================================================
# Orquestación: build_run() y train_run() pensados para notebook
# ============================================================

@dataclass
class Run:
    device: torch.device
    gpt_config: Dict[str, Any]
    settings: Dict[str, Any]
    tokenizer: CharTokenizer
    train_loader: DataLoader
    val_loader: DataLoader
    model: GPTModel
    optimizer: torch.optim.Optimizer
    run_dir: str
    checkpoint_path: str
    perf: Optional[PerfMonitor]
    sysmon: Optional[GPUSystemMonitor]


def _default_gpt_config() -> Dict[str, Any]:
    return {
        "context_length": 256,
        "emb_dim": 512,
        "n_heads": 8,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        # "vocab_size" se completa luego de tokenizar
        "device": pick_device(),
    }


def _default_settings() -> Dict[str, Any]:
    return {
        "learning_rate": 5e-4,
        "batch_size": 16,
        "weight_decay": 0.1,
        "seed": 20,
    }


def build_run(
    data_path: str = "data/input.txt",
    gpt_config: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    perf_cfg: Optional[PerfConfig] = None,
    run_name: Optional[str] = None,
    csv_path: Optional[str] = "./perf_logs/train_metrics.csv",
    enable_sysmon: bool = True,
) -> Run:
    """Construye todo (datos, modelo, optimizer, monitor) pero **no** entrena aún."""
    gpt_config = {**_default_gpt_config(), **(gpt_config or {})}
    settings   = {**_default_settings(),   **(settings or {})}

    set_seed(int(settings.get("seed", 123)))
    device = pick_device()

    # TF32 en CUDA para speed-up
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Datos
    text, tokenizer, data = load_text_and_tokenize(data_path)
    gpt_config["vocab_size"] = tokenizer.vocab_size

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    train_loader, val_loader = make_loaders(train_data, val_data, block_size=gpt_config["context_length"], batch_size=settings["batch_size"])

    # Modelo + optim
    model = GPTModel(gpt_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])

    # Run dir (TensorBoard)
    base_tb = os.path.abspath("./tb_traces")
    os.makedirs(base_tb, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_tb, (run_name or stamp))
    os.makedirs(run_dir, exist_ok=True)

    # PerfMonitor (si existe perfkit)
    perf = None
    if PerfMonitor is not None and PerfConfig is not None:
        perf = PerfMonitor(
            model, device,
            perf_cfg or PerfConfig(
                log_every=100, grad_norm_every=500, warmup_steps_ignore=50,
                enable_tensorboard=True, tb_logdir=run_dir, csv_path=csv_path, dtype_bytes=2,
            )
        )
        # Estimación de memoria (opcional pero útil)
        try:
            # usamos el dataset de train con batch_size settings["batch_size"]
            cal_batch = get_calibration_batch(CharDataset(train_data, gpt_config["context_length"]), settings["batch_size"])
            mem_report = perf.estimate_memory_budget(
                sample_batch=cal_batch,
                emb_dim=gpt_config["emb_dim"], n_layers=gpt_config["n_layers"], seq_len=gpt_config["context_length"],
            )
            print("MemBudget:", {k: v for k, v in mem_report.items() if k != "predict_peak_bytes"})
        except Exception as e:
            print(f"[mem] calibración de memoria falló: {e}")

    # SysMon (telemetría GPU) si hay writer de TB
    sysmon = None
    if enable_sysmon and perf is not None and hasattr(perf, "_tb") and getattr(perf, "_tb") is not None and GPUSystemMonitor is not None:
        try:
            sysmon = GPUSystemMonitor(
                tb_writer=getattr(perf, "_tb", None),
                tb_logdir=run_dir, device_index=0, period_sec=1.0,
                get_step_fn=lambda: perf.global_step,
            )
            sysmon.start()  # no falla si no hay CUDA
        except Exception as e:
            print(f"[sysmon] no se pudo iniciar: {e}")

    checkpoint_path = get_checkpoint_filename(gpt_config)

    return Run(
        device=device,
        gpt_config=gpt_config,
        settings=settings,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        perf=perf,
        sysmon=sysmon,
    )


def train_run(
    run: Run,
    num_epochs: int = 1,
    eval_freq: int = 100,
    eval_iter: int = 10,
    max_steps: Optional[int] = 1000,
    save_chkpt: bool = True,
) -> TrainMetrics:
    metrics = train(
        model=run.model,
        train_loader=run.train_loader,
        val_loader=run.val_loader,
        optimizer=run.optimizer,
        device=run.device,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        max_steps=max_steps,
        checkpoint_path=run.checkpoint_path,
        save_chkpt=save_chkpt,
        monitor=run.perf,
    )
    # flush/close
    if run.sysmon is not None:
        try: run.sysmon.stop()
        except Exception: pass
    if run.perf is not None:
        try: run.perf.close()
        except Exception: pass
    return metrics


# ============================================================
# Front-end simple: main() para una sola llamada desde notebook
# ============================================================

def main(
    data_path: str = "data/input.txt",
    gpt_config: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    num_epochs: int = 1,
    eval_freq: int = 100,
    eval_iter: int = 10,
    max_steps: Optional[int] = 1000,
    run_name: Optional[str] = None,
    save_chkpt: bool = True,
) -> Tuple[GPTModel, CharTokenizer, TrainMetrics, Run]:
    """Conveniencia: construye todo y entrena. Devuelve (model, tokenizer, metrics, run)."""
    run = build_run(data_path=data_path, gpt_config=gpt_config, settings=settings, perf_cfg=None, run_name=run_name,
                    csv_path="./perf_logs/train_metrics.csv", enable_sysmon=True)
    metrics = train_run(run, num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter, max_steps=max_steps, save_chkpt=save_chkpt)
    return run.model, run.tokenizer, metrics, run


__all__ = [
    # datos
    "CharTokenizer", "CharDataset", "load_text_and_tokenize", "make_loaders", "get_calibration_batch",
    # modelo
    "GPTModel", "MultiHeadAttention", "FeedForward", "TransformerBlock",
    # checkpoints / inferencia
    "get_checkpoint_filename", "save_checkpoint", "load_checkpoint", "load_model_for_inference",
    # loss / eval / profiler
    "calc_loss_batch", "calc_loss_loader", "evaluate_model", "run_profiler_window",
    # entrenamiento
    "TrainMetrics", "train",
    # orquestación
    "Run", "build_run", "train_run", "main",
]