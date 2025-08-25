"""
gpt_pretraining_baseline.py

Entrypoint para preentrenamiento de un modelo GPT pequeño, usando PyTorch.
Este archivo está organizado para ser ejecutado como script, no como notebook.

Estructura:
- Imports y configuración
- Tokenizer y Dataset
- Definición del modelo (MultiHeadAttention, TransformerBlock, GPTModel)
- Funciones utilitarias (checkpoint, generación, evaluación, entrenamiento)
- Entrypoint principal (main)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# =========================
# Configuración
# =========================

# gpt_config = {
#     "context_length": 256,
#     "emb_dim": 512,
#     "n_heads": 8,
#     "n_layers": 12,
#     "drop_rate": 0.1,
#     "qkv_bias": False,
#     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# }

# settings = {
#     "learning_rate": 5e-4,
#     "batch_size": 16,
#     "weight_decay": 0.1,
#     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     "seed": 20,
# }

def get_checkpoint_filename(cfg):
    # Crear directorio si no existe
    os.makedirs("model_checkpoints", exist_ok=True)
    return (
        f"model_checkpoints/gpt_checkpoint_ctx{cfg['context_length']}_emb{cfg['emb_dim']}_"
        f"heads{cfg['n_heads']}_layers{cfg['n_layers']}_tokchar.pth"
    )

# =========================
# Tokenizer y Dataset
# =========================

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        if len(data) <= block_size:
            raise ValueError(f"Data length ({len(data)}) must be > block_size ({block_size})")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

# =========================
# Modelo
# =========================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg["device"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    # util simple
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
# =========================
# Benchmarking y métricas
# =========================

# FLOPs pico de GPU (ajustar según tu GPU)
GPU_FP32_PEAK_FLOPS = 26.7e12     # RTX 4000 Ada ~26.7 TFLOPs FP32
GPU_TENSOR_PEAK_FLOPS = 327.6e12  # RTX 4000 Ada ~327.6 TFLOPs (bf16/fp16 tensor cores)

def flops_per_token_training(gpt_cfg, num_params: int):
    """
    Estima FLOPs útiles por token en entrenamiento.
    Aproximación tipo PaLM para entrenamiento por token:
    phi = 6N + 12 * L * d * T
    """
    L = gpt_cfg["n_layers"]
    d = gpt_cfg["emb_dim"]
    T = gpt_cfg["context_length"]
    N = int(num_params)
    return 6*N + 12*L*d*T

def estimate_mfu(tokens_per_sec, gpt_cfg, num_params, peak_flops=GPU_FP32_PEAK_FLOPS):
    """Calcula Model FLOPs Utilization (MFU)"""
    phi = flops_per_token_training(gpt_cfg, num_params)
    achieved = tokens_per_sec * phi
    return achieved / peak_flops

def params_num_bytes(model):
    """Calcula bytes totales de parámetros del modelo"""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total

def optimizer_state_num_bytes_exact(optimizer):
    """
    Mide exactamente los bytes de estados en optimizer.state (si ya hubo backward).
    Si no hay estados todavía, retorna 0.
    """
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total

def reset_peak_memory_stats(device='cuda'):
    """Resetea estadísticas de memoria pico"""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def memory_report(device='cuda'):
    """Imprime reporte detallado de memoria CUDA"""
    if device != 'cuda' or not torch.cuda.is_available():
        return {}
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev)
    reserv = torch.cuda.memory_reserved(dev)
    peak_a = torch.cuda.max_memory_allocated(dev)
    peak_r = torch.cuda.max_memory_reserved(dev)
    
    print(f"[CUDA mem] allocated: {alloc/1e6:.1f} MB | reserved: {reserv/1e6:.1f} MB | "
          f"peak_alloc: {peak_a/1e6:.1f} MB | peak_reserved: {peak_r/1e6:.1f} MB")
    
    return {
        'allocated_mb': alloc/1e6,
        'reserved_mb': reserv/1e6,
        'peak_alloc_mb': peak_a/1e6,
        'peak_reserved_mb': peak_r/1e6
    }

# =========================
# Funciones utilitarias
# =========================

def save_checkpoint(model, optimizer, run, global_step, filename):
    checkpoint = {
        'run': run,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename} (run {run}, step {global_step})")

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        try:
            checkpoint = torch.load(filename, map_location=model.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            run = checkpoint.get('run', checkpoint.get('epoch', -1))  # Compatibilidad
            global_step = checkpoint.get('global_step', -1)
            print(f"Checkpoint loaded from {filename}, resuming from run {run + 1}, step {global_step + 1}")
            return run, global_step
        except Exception as e:
            print(f"Error loading checkpoint {filename}: {e}")
            print("Starting training from scratch")
            return -1, -1
    else:
        print(f"No checkpoint found at {filename}, starting from scratch")
        return -1, -1

def load_model_for_inference(gpt_config, filename, device):
    model = GPTModel(gpt_config).to(device)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filename} for inference")
    else:
        print(f"No checkpoint found at {filename}, returning untrained model")
    return model

def generate_text_simple(model, idx, max_new_tokens, context_size):
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
    encoded = torch.tensor([tokenizer.encode(start_context)], dtype=torch.long, device=device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=max_new_tokens, context_size=context_size
        )
        decoded_text = tokenizer.decode(token_ids[0].tolist())
    model.train()
    try:
        from IPython.display import display, Markdown
        display(Markdown(f"**Generated Output:**\n\n{decoded_text}"))
    except ImportError:
        print(f"Generated Output:\n{decoded_text}")
    return decoded_text

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train(model, train_loader, val_loader, optimizer, device, config, num_runs, eval_freq, eval_iter, 
                      max_steps=None, checkpoint_path=None, save_chkpt=False, 
                      flops_fn=None, peak_flops=GPU_FP32_PEAK_FLOPS, memory_freq=None):
    """
    Función de entrenamiento genérica con métricas detalladas.
    
    Args:
        flops_fn: Función que calcula FLOPs por token (config, num_params) -> int. Si None, no calcula MFU
        peak_flops: FLOPs pico de la GPU para calcular MFU
        memory_freq: Cada cuántos steps mostrar memoria (None = solo al final)
    """
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0
    cumulative_tokens, cumulative_time = 0.0, 0.0
    start_run = 0
    
    # Métricas del modelo
    num_params = model.get_num_params()
    param_bytes = params_num_bytes(model)
    
    print(f"\n{'='*60}")
    print(f"MODEL METRICS")
    print(f"{'='*60}")
    print(f"#params: {num_params:,} ({num_params/1e6:.2f} M)")
    print(f"Parámetros (bytes): {param_bytes/1e6:.1f} MB")
    
    # Resetear memoria pico antes de empezar
    reset_peak_memory_stats(device)
    
    # Cargar checkpoint si existe
    if checkpoint_path is not None:
        loaded_run, loaded_global_step = load_checkpoint(model, optimizer, checkpoint_path)
        if loaded_run >= 0:  # Si se cargó exitosamente
            start_run = loaded_run + 1
            global_step = loaded_global_step
            print(f"Resuming training from run {start_run}, global step {global_step + 1}")
            
            # Ajustar max_steps si el checkpoint ya pasó el límite
            if max_steps is not None and global_step >= max_steps:
                original_max_steps = max_steps
                max_steps = global_step + original_max_steps
                print(f"Checkpoint is at step {global_step + 1}, extending training for {original_max_steps} more steps (until step {max_steps})")
    
    print(f"\n{'='*60}")
    print(f"TRAINING STARTED")
    print(f"{'='*60}")
    
    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_start.record()
    else:
        t0 = time.time()
    training_finished = False
    for run in range(start_run, start_run + num_runs):
        if training_finished:
            break
        model.train()
        for inp_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1
            loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            loss.backward()
            optimizer.step()
            total_tokens += inp_batch.numel()
            if global_step % eval_freq == 0:
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()
                    elapsed = t_start.elapsed_time(t_end) / 1000
                    t_start.record()
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()
                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / elapsed if elapsed > 0 else 0
                if global_step:
                    cumulative_tokens += tokens_interval
                    cumulative_time += elapsed
                avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)
                
                # Calcular MFU si se proporciona función de FLOPs
                mfu_str = ""
                if flops_fn is not None:
                    mfu = estimate_mfu(tps, config, num_params, peak_flops)
                    mfu_str = f", MFU: {mfu*100:.1f}%"
                
                print(f"Run {run+1}, Step {global_step:06d}, "
                      f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                      f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}{mfu_str}")
                
                # Reporte de memoria periódico si se especifica
                if memory_freq and global_step % memory_freq == 0:
                    memory_report(device)
                if max_steps is not None and global_step >= max_steps:
                    print(f"Reached max steps: {max_steps}")
                    if save_chkpt and checkpoint_path:
                        save_checkpoint(model, optimizer, run, global_step, checkpoint_path)
                        print(f"Final checkpoint saved at max_steps")
                    training_finished = True
                    break
        if save_chkpt:
            save_checkpoint(model, optimizer, run, global_step, checkpoint_path)
    
    # Reporte final completo
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED - FINAL METRICS")
    print(f"{'='*60}")
    
    # Métricas de optimizer state
    opt_bytes = optimizer_state_num_bytes_exact(optimizer)
    if opt_bytes > 0:
        print(f"Optimizer state (bytes): {opt_bytes/1e6:.1f} MB")
    
    # Timing final
    if cumulative_time > 0:
        final_tps = cumulative_tokens / cumulative_time
        print(f"Avg step time: {(cumulative_time * 1000 / max(1, global_step + 1)):.2f} ms")
        print(f"Tokens/s: {final_tps:,.0f}")
        
        # MFU final
        if flops_fn is not None:
            final_mfu = estimate_mfu(final_tps, config, num_params, peak_flops)
            print(f"MFU (peak = {peak_flops/1e12:.1f} TFLOPs): {final_mfu*100:.2f}%")
    
    # Reporte final de memoria
    memory_report(device)
    print(f"{'='*60}")
    
    return train_losses, val_losses, track_tokens

# =========================
# Entrypoint principal
# =========================

def run(gpt_config, settings, train_model=True, max_steps=None, num_runs=1, 
                           eval_freq=100, save_checkpoint=True, load_existing=True,
                           flops_fn=flops_per_token_training, peak_flops=GPU_FP32_PEAK_FLOPS, memory_freq=50):
    """
    Versión avanzada de run() con control total sobre las métricas.
    
    Args:
        flops_fn: Función personalizada para calcular FLOPs (config, num_params) -> int
                 Si None, no se calcula MFU
        peak_flops: FLOPs pico de tu GPU para MFU
        memory_freq: Cada cuántos steps mostrar memoria (None = solo al final)
    
    Examples:
        # Con MFU usando función por defecto
        run(config, settings, max_steps=100)
        
        # Sin MFU (más rápido)
        run(config, settings, max_steps=100, flops_fn=None)
        
        # Con memoria cada 50 steps
        run(config, settings, max_steps=100, memory_freq=50)
        
        # GPU diferente
        run(config, settings, peak_flops=40e12)  # RTX 4090
    """
    # Usar misma lógica que run() pero con parámetros personalizados
    torch.manual_seed(settings["seed"])
    device = settings["device"]
    gpt_config = gpt_config.copy()
    gpt_config["device"] = device

    # Load data
    input_path = "data/input.txt"
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    gpt_config["vocab_size"] = vocab_size

    checkpoint_path = get_checkpoint_filename(gpt_config)
    model = GPTModel(gpt_config)
    model.to(device)
    
    train_losses, val_losses = [], []
    
    if train_model:
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        train_dataset = CharDataset(train_data, gpt_config["context_length"])
        val_dataset = CharDataset(val_data, gpt_config["context_length"])
        train_loader = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"]
        )

        train_losses, val_losses, _ = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            config=gpt_config,
            num_runs=num_runs,
            eval_freq=eval_freq,
            eval_iter=1,
            max_steps=max_steps,
            checkpoint_path=checkpoint_path if load_existing else None,
            save_chkpt=save_checkpoint,
            flops_fn=flops_fn,
            peak_flops=peak_flops,
            memory_freq=memory_freq
        )
    else:
        if load_existing and os.path.isfile(checkpoint_path):
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
            run, global_step = load_checkpoint(model, optimizer, checkpoint_path)
            print(f"Model loaded from checkpoint (run {run+1}, step {global_step+1})")
        else:
            print("Using untrained model")
    
    return model, tokenizer, device
