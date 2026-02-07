# AI notas

https://github.com/lucaspecina/ai-basics

https://github.com/lucaspecina/ai-foundations

**AI sin humo**
Minimizar el esfuerzo para aprender

Fuentes importantes:

- understanding deep learning book
- why machines learn book
- karpathy videos y codigo (micrograd, transformers, etc)
- sebastian raschka book LLMs
- Huyen book LLMs
- https://www.youtube.com/watch?v=NrO20Jb-hy0&list=LL&index=16&t=634s&ab_channel=WelchLabs
- https://github.com/vukrosic/gpt5-from-scratch/blob/main/gpt5_from_scratch.ipynb

---

---

# Aprendizaje supervisado

- Tipos: supervisado, RL, no supervisado, auto supervisado
- Supervisado: idea y funcionamiento
    - Idea de funcion con input y parametros (ejemplo linear function)
    - Clasificacion y regresion
    - Dataset, target, loss, optimization (general)
    - (Basico) Familias de funciones aprendibles: arboles, gradient boosting, redes neuronales (general)
- Overfitting, bias-variance trade-off

---

Cuando se dice aprender es simplemente tener un modelo, una funcion parametrica en general, y usar datos provenientes de una distribucion para inferir los parametros del modelo, de una forma estadistica.

CASI SIEMPRE que hablamos de aprendizaje, lo que queremos en el fondo es hacer una prediccion sobre algo: un numero, una clase binaria, una clase con muchos posibles resultados, etc.

**Tipos**:

**Supervisado**:

Lo que hacemos es tener dos cosas: Datos de entrada (variables predictoras) y un target. Tenemos el concepto de instancia, que es un ejemplar de algo, por ej una casa, con sus caracteristicas y una de las caracteristicas es la que a nosotros nos interesa predecir.

Lo que queremos hacer es crear una FUNCION (una ecuacion matematica) que mapee de los inputs a los outputs. Entonces cuando le mandamos nuevos inputs, nos devuelve un output (inferencia). Esa funcion tiene parametros, y el valor de esos parametros cambia el resultado de la funcion. Entonces tenemos un tipo de modelo, que es una familia de modelos finales (porque dependen de los parametros elegidos).

Entonces cuales deberian ser el valor de esos parametros? Eso es lo que se llama TRAINING o learning. La idea es que, como ya tenemos pares de inputs → outputs que sabemos que estan bien mapeados, la idea es encontrar valores de los parametros que dados esos inputs den los resultados mas parecidos a los outputs que ya conocemos.

Intuicion: pensar siempre que los datos viven en un espacio multi dimensional, y el target tambien, puede ser de una clase o un numero mas… pero que siempre vivimos en un espacio donde todo lo que tenemos es un punto en ese espacio o un vector. Lo que queremos es encontrar y determinar que zonas de ese espacio son de una clase o de la otra, o cual es la linea o plano que se acercan mas a los puntos.

Cada *modelo* es una función que dibuja una forma (una línea, un plano, una superficie curva) que intenta *separar*, *ajustarse a* o *aproximar* esos puntos.

Ejemplo: linear regression 1D: tenemos un input x de una dimension y un output y de una dimension. O sea, tenemos un espacio de dos dimensiones. 

![image.png](AI%20notas/image.png)

Vamos a tratar de modelarlo con una familia LINEAL de modelos: 
y = f(x, p) = p1 + x * p2. Con esta familia, podemos armar cualquier recta, dependiendo de los valores de los parametros: el slope y el intercept.

![image.png](AI%20notas/image%201.png)

Como hacemos para encontrar la mejor recta? Podemos definir una FUNCION DE PERDIDA. Esto es, una funcion que queramos OPTIMIZAR (minimizar en este caso) para acercarnos a la funcion que nosotros queremos.

Podemos tener cualquier tipo de funcion de perdida. Podria ser la suma o el promedio de las diferencias entre las predicciones y los puntos reales… 

Se suele usar el least squared loss o mean squared loss.

Loss(p) = SUMA( yhat - y ) **2 … o … Loss(p) = MEAN( yhat - y ) **2

Se suele hacer al cuadrado porqeu tiene buenas propiedades: Deja todos las perdidas individuales en positivo, si no se podrian compensar, tambien penaliza mas a los que mas distantes estan (los errores grandes pesan mas)… y tambien es diferenciable y es convexa (si no tambien es convexo pero no diferenciable).

![image.png](AI%20notas/image%202.png)

Ahora tenemos un nuevo espacio: El de la LOSS (las coordenadas son los parametros):

![image.png](AI%20notas/image%203.png)

Esa es la forma que se genera en la loss. Entonces tenemos una funcion diferenciable en todos lados, que tenemos que MINIMIZAR para encontrar la menor perdida (que es la diferencia entre los valores reales y lo predicho). 

Esa es la tarea del APRENDIZAJE SUPERVISADO.

Como lo hacemos? Con un concepto llamado gradient descent. El proceso de bajar por esa funcion hasta el punto minimo o cercano es lo que se llama ENTRENAMIENTO y para hacerlo usamos los datos que ya tenemos.

![image.png](AI%20notas/image%204.png)

Ejemplo 2: regresion logistica

Hasta ahora vimos un problema de regresión: predecir un número real.

Ahora vamos con otro tipo de predicción muy común: **decidir una clase**.

Por ejemplo: ¿Este mail es *spam* o *no spam*? ¿Esta imagen es de un *gato*, *perro* o *caballo*? ¿Este usuario va a *comprar* o *no comprar*?

Este tipo de problema se llama **clasificación**. Si hay solo dos clases posibles (como sí o no), se llama **clasificación binaria**. Seguimos con la misma idea:

> Queremos encontrar una función con parámetros que, dados los datos de entrada, prediga a qué clase pertenece ese punto.
> 

Pero ahora no nos interesa predecir un número como 37.2 Sino **la probabilidad** de que pertenezca a una clase u otra.

Se llama regresión logística porque sigue teniendo forma parecida a una recta, pero transformada para que su resultado esté entre 0 y 1. → sigmoid function → transforma los resultados entre 0 y 1 en forma de S.

- Si el resultado es cercano a **1**, el modelo cree que el ejemplo es de la clase positiva.
- Si es cercano a **0**, cree que es de la clase negativa.
- Si está cerca de **0.5**, no está muy seguro.

Loss:

Ahora no nos sirve usar la diferencia al cuadrado, como hacíamos en regresión porque hay varios problemas con gradientes poco informativos, etc…

Ahí entra la función de pérdida más usada en clasificación binaria: **CROSS-ENTROPY (LOG LOSS)**

Para una muestra es: - ( y * log(p)  +  (1-y) * log(1-p) )

- Si la etiqueta real y=1y = 1y=1, la pérdida es −log⁡(p)
    - Si p=0.9, entonces  −log⁡(0.9)≈0.105. Pequeño castigo porque estuvo “casi seguro y acertó”.
    - Si p=0.1, entonces  −log⁡(0.1)≈2.302. Castigo grande porque estuvo seguro y falló.
- Si la etiqueta real y=0, la pérdida es −log⁡(1−p)
    - Si p=0.1,  −log⁡(0.9)≈0.105. Castigo pequeño (acertó).
    - Si p=0.9,  −log⁡(0.1)≈2.302. Castigo grande (falló).

Este comportamiento “penaliza más fuerte” las predicciones muy confiadas y equivocadas, lo que ayuda a que el modelo ajuste sus pesos con gradientes más sólidos cuando más se lo necesita.

Ejemplo multiclase (por ej LLM):

Ahora tenemos K clases posibles (tipos de flores o tokens en el vocabulario para un LLM).

La salida del modelo genera un valor para cada uno que no es una probabilidad, pero despues se pasa por un softmax para pasarlo a probabilidad.

Loss: 

Si la etiqueta verdadera es la clase t∈{1,2,…,K}, podemos representarla como un vector “one-hot”:

La cross entropy para una sola muestra es: - SUMA(de i a K) y_i * log(p_i)… pero como estamos haciendo one hot, donde todo se va a multiplicar por 0 menos 1 solo valor, es lo mismo que hacer: -log(p) para la clase correcta.

Por que log?

Porque estamos hablando de probabilidades, por lo que es bastante beneficioso usar log, porque funciona como una sorpresa (TODO: VER MAS TARDE)

![image.png](AI%20notas/image%205.png)

- Por que no usar cross entropy en regresion entonces?
    
    ```markdown
    Aunque la predicción esté acotada a \[0, 1], **si el “target” es un valor continuo**, BCE (-log p) no captura bien el error de distancia numérica.
    
    * **BCE asume que el target es la probabilidad de una etiqueta binaria** (0 o 1), o como “etiqueta difusa” en \[0, 1] interpretada como probabilidad.
    
      * Ejemplo: si el target fuese 0.7, BCE lo trataría como “70 % de probabilidad de etiqueta 1” en un Bernoulli, no como “el número 0.7 exacto” que queremos predecir.
    
    * **Usar BCE con target continuo tiende a forzar la predicción a valores “extremos” (0 o 1)** en vez de acercarla linealmente a 0.7. Eso hace que:
    
      1. Si predices 0.5 y el target es 0.7, BCE = $-[0.7·\log(0.5)+0.3·\log(0.5)]\approx0.69$.
      2. Si predices 0.9 para ese mismo target 0.7, BCE = $-[0.7·\log(0.9)+0.3·\log(0.1)]\approx1.20$ (peor), aunque 0.9 está “numéricamente más cerca” de 0.7 que 0.5.
    
    Con **MSE** en cambio:
    
    * Si target=0.7 y tú predices 0.5, MSE=(0.7–0.5)²=0.04.
    * Si predices 0.9, MSE=(0.7–0.9)²=0.04.
      Ambas predicciones equidistan de 0.7 y reciben el mismo castigo, reflejando la distancia real.
    
    **Conclusión:**
    
    * BCE (–log p) “mide sorpresa” de eventos binarios o probabilidades de etiquetas, no la cercanía de números continuos.
    * Aunque la salida esté en \[0, 1], si el objetivo es un valor real, usa MSE (o MAE). Si quisiéramos forzar un modelo probabilístico con datos en \[0, 1], habría que usar una distribución continua apropiada (por ejemplo, Beta loss), pero no BCE.
    ```
    

# Perceptron

- Funcion lineal
- Funcion basica lineal con decision boundary para clasificacion
- Espacio 2d de los parametros y los inputs y como se arma el boundary (comparacion con dot product)
- Gradient descent (basico) para encontrar los valores de los parametros
<Buscar si se hace asi en perceptron vs el algoritmo de Rosenblatt>

---

Que pasa si a partir de ella queremos aplicar algo de logica, como OR, AND, NOT, etc.

Partimos de la funcion lineal porque es una de las funciones mas basicas existentes para modelar algo.

Podemos agarrar inputs y sumarlos, y si pasa un umbral, se clasifica de una manera o de otra. Se suele usar clasificacion binaria para las primeras pruebas porque se quiere probar con logica booleana, que es binaria.

Activacion: 

Se usa la activacion para SACARLE LO LINEAL a la funcion lineal. Lo mas basico es hacer un umbral y si es menor a 0, hacerlo -1 y si es mayor hacerlo 1. Entonces ahi hicimos una funcion no lineal basica que puede servir para clasificacion.

Hay otras, como la sigmoide, que transforma todo el espacio entre 0 y 1, y puede servir tambien para modelar probabilidades.

**Perceptron (funcion lineal + activacion)**:  z = x * w + b, { si z<0 → y=-1 , otherwise y=1 }

![image.png](AI%20notas/image%206.png)

Para un input solo x, puede tomar cualquier valor, y va a quedar de un lado o del otro de la clasificacion segun el valor de w y de b, mas la activacion (umbral en 0). Entonces si w=1 y b=1, si tenemos x=1 → z=2, por lo tanto y=-1. 
Para que sea clasificado como 1, x deberia ser < -1. 

Con inputs 2D es lo mismo.

![image.png](AI%20notas/image%207.png)

Como pensar en esta frontera? Se puede pensar que w (en este caso es 2d) es un vector y la frontera es ortogonal a ese vector. Entonces mientras un punto este mas alineado al vector w, mas grande va a ser su valor y por lo tanto mas lo va a clasificar como 1. Esto es lo mismo que pensar en el Dot product.

Dot product: cuando tenemos dos vectores, multiplicamos cada uno de los componentes de uno con el mismo componente del otro y despues sumamos todo y nos queda solo un escalar. Esto nos dice que tan ALINEADO estan estos dos. Como es una multiplicacion, si los dos son altos, va a pesar mucho en el final… Si uno un poco si y el otro no, no suma tanto. Si los dos son bajos, no aporta casi nada. Es una forma de medir lo que es importante para ambos vectores.

![image.png](AI%20notas/image%208.png)

El bias empuja el hiperplano para un lado o el otro segun cuanto valga, siempre en linea con el vector w.

TODO: completar un poco mas esto

Problema XOR: hay cosas que no pueden ser resueltas por un simple perceptron.

![image.png](AI%20notas/image%209.png)

Porque el perceptron puede clasificar todo lo que sea linearmente separable. Un XOR no es linearmente separable por lo que no puede resolverlo.

# Shallow networks

Lineal: Como tenemos un perceptron podemos tener dos y eso generaria dos rectas pero al combinarlas nos quedaria una funcion lineal igual porque combinar dos lineales da una lineal. Entonces tampoco lo resolveriamos.

Con activacion: incluso usando un threshold simple ya podemos resolverlo. 

La idea es: tenemos dos perceptrones que cada uno recibe el input inicial. Cada uno es una funcion no lineal porque tiene un threshold… entonces despues al combinarlos, nos da otra funcion no lineal que lo puede resolver.

![image.png](AI%20notas/image%2010.png)

En forma de pesos y sesgos:

- w_1 = (1,1), b_1 = -0.5
- w_2 = (1,1), b_2 = -1.5
- w_out = (1, -1), b_out = -0.5

En este caso, LE ESTAMOS PONIENDO ACTIVACION FINAL threshold al output total del modelo, para que quede binario para clasificacion… pero podriamos no usarlo y que quede crudo despues de la combinacion (y despues clasificarlo con un threshold nuestro, aparte). Lo importante es entender que al agregar no linealidad a las hidden units, estamos partiendo el espacio y quebrandolo y dandole potenciales formas que hacen a la funcion mas expresiva.

Que se use el threshold como activacion no implica que la salida final del modelo siempre sea binaria. Al combinarlos, puede dar otros valores...

![image.png](AI%20notas/image%2011.png)

El problema es que no es diferenciable y aparte que perdes informacion al ser binaria en cada neurona, no capta ciertas cosas porque es menos expresiva, por eso por lo general usamos otra como ReLU o sigmoid

Ademas no es smooth, no tiene rampas ni inclinaciones. Siempre se mueve de a saltos. Y no es que se pueda suavizar porque si le agregas algo al final, como sigmoide, va a transformar todo en sigmoide, deberia hacerse en cada unit.

![image.png](AI%20notas/image%2012.png)

**Hay que recordar que todo lo que queremos hacer siempre es particional el espacio de inputs y asignarle valores**, que representen algo: una decision (0,1), una probabilidad, un valor numerico. Mientras mas particionemos, mas detallado es.

La forma de particional el espacio de inputs para asignarle valores distintos es **agregandole no-linealidades (activacion) a las neuronas**, porque eso particiona el espacio y, al combinarlas despues, nos da flexibilidad para que cada particion tenga valores o comportamientos distintos.

Otras activaciones (no linealidades): queremos que sean diferenciables o que puedan aproximarse a diferenciables y queremos que sean expresivas (por eso binarias se quedan cortas) y aparte SMOOTH, suaves (las binarias, en el resultado final de la red siempre pasan de un valor al otro con saltos, no hay rampas ni inclinaciones).

- ReLU: Simple, si el valor es menor a 0, convertirlo en 0… si es mayor, dejarlo igual.
- Sigmoid: deja valores entre 0 y 1 con una funcion de forma de S (tiene parametros).

ReLU

![image.png](AI%20notas/image%2013.png)

**Shallow networks**

```markdown
        x ──► [ w₁·x + b₁ ] ──► ReLU ──► h₁ ┐
           ──► [ w₂·x + b₂ ] ──► ReLU ──► h₂ ├─► (combinación lineal) ──► y
           ──► [ w₃·x + b₃ ] ──► ReLU ──► h₃ ┘
```

En vez de hacerlo secuencial:

```bash
x₁ → [Perceptrón 1 (w₁,b₁)] → ReLU → [Perceptrón 2 (w₂,b₂)] → ReLU → [Perceptrón 2 (w₂,b₂)] → ReLU → y
```

Porque en este caso secuencial se va delimitando el espacio JERARQUICAMENTE, por lo que va matando zonas y solo trabajamos para hacer particiones consecutivas solo en el rango que quedo vivo, no en todo el espacio.

Si queremos usar todo el espacio y cortarlo de distintas manera, no jerarquicamente, podemos hacer una CAPA de neuronas con ReLU (en paralelo) y despues simplemente combinarlas linealmente.

Shallow network tiene solo 1 hidden layer

- **Neurona**: hidden unit
- **Layer**: conjunto de neuronas que se hacen en paralelo (perceptrones con activacion no lineal)

**input layer (datos de entrada) → hidden layer (multiples neuronas) → output layer (combinacion lineal final)**

![image.png](AI%20notas/image%2014.png)

![image.png](AI%20notas/image%2015.png)

Ejemplo con ReLU:

- pre-activacion: son las funciones lineales de cada hidden unit (cada neurona en la hidden layer). Aca son 3.
- activacion: es pasarle el ReLU a las activaciones y generar el output de cada hidden unit
- escalado por pesos: es la salida de cada hidden unit pesado por el weight correspondiente hacia el output
- salida final: es la suma de los outputs de las hidden units escalados + el bias del output

![image.png](AI%20notas/image%2016.png)

**Particionamiento del espacio en regiones**

Como vimos muchas veces, la clave es particionar el espacio en muchas regiones. Con ReLU podemos particionarlo en muchas regiones lineales que sean continuas, donde vaya cambiando la pendiente.

Ya vimos que con threshold simple no crea regiones lineales continuas sino saltos o steps. Con sigmoid no se crean regiones lineales sino algo mas continuo, como curvas suaves.

![image.png](AI%20notas/image%2017.png)

**Dejamos de lado la funcion de activacion threshold** porque no es derivable y mas adelante no la vamos a poder usar para entrenar los modelos.

**Regiones lineales con ReLU**

Cada particion esta dada por una hidden unit porque es la que hace la activacion (la no linealidad). Entonces al generar varios cortes, despues al combinar los cortes, nos quedan varias regiones lineales. 

Cuantas? Depende de dos cosas:

- cantidad de neuronas en la hidden layer
- dimension del input

Para input 1D (un solo x), tenemos como maximo n+1 regiones lineales (n es la cantidad de hidden units). Cada hidden unit aporta un corte en el espacio y al combinarlos se crean n+1 (puede ser menos si el corte es en el mismo lugar en varias hidden units).

En inputs 2D, se corta con un hiperplano, y si hay mas espacio, se generan mas particiones.

![image.png](AI%20notas/image%2018.png)

1. Con input 1D, 1 hidden unit crea una unión, que divide el eje en dos regiones lineales.
2. Con input 2D, dos hidden units puede dividir el espacio de entrada usando dos líneas (aquí alineadas con los ejes) para crear cuatro regiones.
3. Con input 3D, 3 hidden units puede dividir el espacio de entrada usando tres planos (nuevamente alineados con los ejes) para crear ocho regiones.

Input dimension D y D hidden units → 2^D regiones lineales

![image.png](AI%20notas/image%2019.png)

![image.png](AI%20notas/image%2020.png)

Resolver XOR con ReLU (2 hidden units)

![image.png](AI%20notas/image%2021.png)

Resolver XOR con sigmoid (2 hidden units)

![image.png](AI%20notas/image%2022.png)

Aca no estamos aplicando funcion de activacion final sino que estamos dejando el output de la red crudo… si quisieramos clasificar, podriamos ponerle un threshold y que lo transforme en 0 o 1.

**Aproximador universal** (shallow network)

Cualquier funcion continua que queramos, no importa que tan compleja sea, puede ser aproximada arbitrariamente bien por una shallow network de solo 1 hidden layer, si tiene las suficientes neuronas (hidden units) y una activacion no lineal (pero no se incluye threshold porque no es una activacion continua y entonces no garantiza una aproximacion uniforme. No lo puede aproximar suavemente).

1. Vos querés una función que haga "algo" con los inputs: prediga un precio, una probabilidad, una clase, etc.
2. Esa función puede tener curvas, quiebres, subidas y bajadas.
3. El teorema te dice: Con **una sola capa oculta**, pero **muchas neuronas**, ya podés construir una función que **se parezca tanto como quieras** a la original.
4. Lo único que necesitás es que la activación **rompa la linealidad**. Por eso no sirve si todo es lineal.

Una funcion continua es la que podes dibujar sin levantar el lapiz del papel. O sea que una con escalon no se puede.

Podemos resolver problemas mas complejos, como clasificar puntos que esten rodeados por puntos de otra clase:

![image.png](AI%20notas/image%2023.png)

Con 8 hidden units y ReLU

![image.png](AI%20notas/image%2024.png)

A medida que aumentamos la cantidad de hidden units, se torna mas flexible para representar la funcion que queramos.

**Multiples inputs y outputs**

Podemos tener muchos inputs y muchos outputs. 

Los outputs pueden representar cualquier cosa que queramos y podemos ponernos creativos con eso. Cada uno puede representar un valor para cosas distintas, por ejemplo en un robot, cuanto se deberia mover el brazo izquierdo y el derecho. Si tenemos multiples outputs, cada uno puede representar una clase de algo (tokens) y su valor se activa mas fuerte cuando esa clase es mas probable (logits).

![image.png](AI%20notas/image%2015.png)

Cuando tenemos 2 outputs, es como que se hacen dos procesos en paralelo, diferentes, que no se hablan entre si, entonces quedan dos funciones o dos valores separadas.

Sin embargo, podemos usar algo mas para combinarlas… por ejemplo, si estamos representando muchas clases y cada uno genera un valor (logit), podemos usar softmax para transformarlos en probabilidades (entre 0 y 1 y que sumen 1 entre todos). Ahi si se estarian combinando y pasarian a ser una unica cosa:

![image.png](AI%20notas/image%2025.png)

Aca el softmax hace que siempre sumen 1 entre ambos valores.

Codigo en ai-utils simple de modelo

# Deep neural networks

- Por que deep… mas eficiencia
- Notacion matricial y codigo basico
- Paso a paso de las transformaciones visual y codigo

---

Ya con las shallow podemos hacer todo lo que queremos, no seria en principio necesario hacer mas nada. Sin embargo, para algunas funciones complejas, la cantidad de neuronas que se necesitan en la hidden layer son MUCHAS. 

Con deep networks (que son composiciones de muchas capas, en vez de una), se pueden crear muchas mas regiones lineales usando menos parametros. Es una cuestion de eficiencia computacional.

Composicion de shallow networks:

Si en vez de terminar con el output lo que hacemos es pasarlo por OTRA shallow network, lo que hacemos es pasar de una dimension de input a otra de output y esto mandarlo a una nueva red shallow.

![image.png](AI%20notas/image%2026.png)

Lo que hacemos aca es que, si tenemos 3 linear regions, para distintos valores de input le puede corresponder el mismo valor de la primera red… eso esta claro. Entonces esos tres valores diferentes de input mapean de la misma manera a un lugar de la segunda network, por lo que si vemos el mapeo desde el input x a la network 2, es como que se triplica todo de una manera escalada. 

Para n=6 units:

- shallow: tenemos como maximo n+1 = 7 linear regions
- compuesta: 4 regiones para la primera red y 4 para la otra. Como para cada region le corresponde la total combinacion con la totalidad de la red 2, se particionan x4, quedando 16 linear regions.

Se puede ver como que para cada region de la red 1 le corresponde toda la transformacion de la red 2, entonces es como doblar el espacio y aplicarle toda la red 2 a cada fold:

![image.png](AI%20notas/image%2027.png)

Es como que se “copia” la red 2 para cada pliegue de la red 1 y se escala segun el tamaño del pliegue de la red 1. No hay que pensarlo como que se multiplican ni nada, sino que se copia la red 2. A su vez, como se dobla el espacio de inputs, la primera parte queda en un sentido, el sentido normal, la segunda queda en el otro sentido (porque se esta dolando para el otro lado, y la ultima queda en el sentido normal. Entonces al aplicar la segunda red, la segunda parte queda como invertida.

**Deep networks**

En realidad vemos que no es necesario “encapsular” todo el resultado de la hidden layer 1 en una neurona “y” y despues pasar eso como input a la otra neurona. Si simplemente conectamos cada neurona de la hidden 2 a cada neurona de la hidden 1 es lo mismo, incluso es una familia mas amplia (TODO: ver bien por que).

![image.png](AI%20notas/image%2028.png)

![image.png](AI%20notas/image%2029.png)

Notacion matricial

![image.png](AI%20notas/image%2030.png)

Las redes son transformaciones lineales y funciones de activacion. Se puede describir: 

- Input: 1D
- Hidden layers: 2
- Hidden units per layer: 3
- Activacion: ReLU

![image.png](AI%20notas/image%2031.png)

**Como implementar redes neuronales**

Codigo simple de una red (en ai-utils)

# Entrenar modelos

- Gradient descent basico, funciones diferenciables
- Backprop, explicacion teorica
- **Loss function** (likelihood y como componer loss functions en general)
- Gradient descent implementacion (optimizers: stochastic, momentum, inicializacion, problemas, inicializacion)
- Performance y Regularizacion (overfitting)
- Implementacion en codigo de backprop con operaciones diferenciables como network (micrograd)
- Implementacion del flujo de entrenamiento total con pytorch (raschka basics, karpathy, udl)
- GPUs para acelerar (raschka) y ARITMETICA (flops, memoria) para inferencia y training

---

Hasta aca vimos la **expresividad de la familia de modelos de redes neuronales**.

Ahora vamos a ver como FITTEARLO A DATA para hacer predicciones.

Esto es una continuacion de supervised learning. Aca vemos toda la estrategia de que significa aprender supervisadamente y los mecanismos para hacerlo y la performance.

Ya sabemos que tenemos:

- Datos x,y
- Modelo expresivo deep network
- Funcion de perdida u objetivo (usa y real y prediccion del modelo para medir error)

Tambien ya sabemos que el modelo depende de los parametros que tiene y que estos pueden modificarse… en una funcion lineal tenemos w, b que segun sus valores, la recta cambia. Con deep networks pasa lo mismo con los weights, bias.

Entonces si queremos que el modelo prediga bien, podemos ajustar los parametros para que el modelo se ajuste a los datos, y para eso podemos usar el truco de MINIMIZAR EL ERROR (la diferencia entre la prediccion y el valor real, eso es la loss function).

## Loss functions

Como pensamos en construir el error? El framework es probabilistico, podemos ver la prediccion del modelo en realidad como probabilidades que tienen los outputs. Queremos siempre aumentar la probabilidad del valor o clase correcta.

![image.png](AI%20notas/image%2032.png)

Para transformar en probabilidad, lo que hacemos es ELEGIR UNA DISTRIBUCION PARAMETRIZADA, para los outputs y… y hacemos que el modelo prediga uno o mas parametros de esa distribucion.

Por ej para un caso de regresion, podemos elegir la distribucion normal y que el modelo prediga la media (y podriamos dejar la varianza como una constante… o tambien predecirla).

MAXIMUM LIKELIHOOD framework: en el entrenamiento tenemos muchos datos, la idea es elegir los parametros que maximicen la probabilidad combinada para todos los ejemplos de entrenamiento.

Esto es la MULTIPLICACION DE LAS PROBABILIDADES de la clase o valor correcto, para cada ejemplo.

Esto es un problema, porque muchas multiplicaciones de probabilidades da un valor muy chiquito, para eso se usa LOG, que transforma en una funcion parecida pero ahora se pueden usar sumas de las log probs. (mas abajo lo vemos mejor). 

Entonces lo que hacemos ahora es MAXIMIZAR EL LOG-LIKELIHOOD… 

o MINIMIZAR EL **NEGATIVE LOG LIKELIHOOD.**

Receta para construir loss functions (con maximum likelihood):

- Elegir una distribucion de prob acorde definida sobre el dominio de predicciones, con parametros
- Hacer que el modelo prediga uno o mas parametros de esa distribucion
- En entrenamiento, buscar parametros que minimicen el negative log likelihood para el conjunto de entrenamiento
- Para la inferencia en nuevos datos, se puede devolver el punto con mayor prob del output

Podemos predecir cualquier cosa, en realidad, multiples outputs. No solo un valor sino cosas re distintas… eso se hace seteando mas outputs para el modelo.

Forma de la loss:

Primero vemos que forma tiene el error de un modelo. Visualizaciones.

La loss function dependen de dos cosas:

- Los parametros del modelo
- La data (un batch genera una loss function distinta a otro batch)

> x afecta cómo varía la predicción yhat con respecto a los pesos → cambia la pendiente del error al mover w.
y afecta dónde está el mínimo de pérdida → cambia el offset del error (el objetivo al que querés llegar).
> 

Aca, con un modelo muy simple con input dimension 1, output dimension 1 y UN SOLO PARAMETRO W, vemos como cambian las funciones con respecto a w, x (data), y (data).
Usamos 4 ejemplos donde a veces dejamos fijo x y otros y (ver coordenadas)

Esto igualmente con respecto a w genera esta loss convexa

![image.png](AI%20notas/image%2033.png)

![image.png](AI%20notas/image%2034.png)

**Regresion (mean squares error loss / least squares error loss)**

Si usamos una **distribucion normal univariada** con el framework de neg log likelihood:

![image.png](AI%20notas/image%2035.png)

Manipulando algebraicamente, llegamos a LSE:

$$
LSE = \sum_i^N (y_i-yhat_i)^2\\
MSE = \frac{1}{N} \sum_i^N (y_i-yhat_i)^2
$$

Ya hablamos de que es conveniente usar el cuadrado, en vez del error comun entre prediccion y valor real porque:

1. Hace todo positivo (si no los errores se pueden compensar)
2. tambien penaliza mas a los que mas distantes estan (los errores grandes pesan mas)
3. tambien es diferenciable y es convexa (si no tambien es convexo pero no diferenciable).

Construccion (modelo simple sin no-linealidad)

![image.png](AI%20notas/image%2036.png)

Ahora con varios ejemplos (en un batch)

![image.png](AI%20notas/image%2037.png)

Ahora agregandole no linealidad en la funcion (tanh)

![image.png](AI%20notas/image%2038.png)

**Clasificacion (cross-entropy / negative log likelihood)**

1. El modelo larga valores… estos valores para cada clase, que no son probabilidades ni nada sino simplemente valores. Esto se llama **LOGITS**. Hay un logit para cada clase.
2. Despues de los logits, lo que se hace es transformarlos en probabilidades, y para eso se usa softmax.
lo usamos porque esperamos K outputs (que son los parametros de nuestra distribucion categorica que elegimos para la clasificacion multiclase) y estos parametros requieren ser entre 0 y 1 y sumar 1 entre todos. Entonces softmax hace cumplir todas esas:

![image.png](AI%20notas/image%2039.png)

1. Una vez que son probabilidades, solamente agarramos la prob de la CLASE CORRECTA.
2. Calculamos el negativo del LOG de esa prob

$$
loss = - log(p)
$$

Son todas las mismas (cross-entropy / log loss / negative log likelihood). La diferencia es que:

- negative log likelihood: espera de inputs ya los probs (el softmax hecho)
- cross entropy: espera los logits crudos y computa el softmax adentro de la formula
- log loss: es para BINARIO tambien espera las probabilidades en forma de sigmoid function (porque cuando tenemos binario tenemos un solo output y para transformarlo en prob no usamos softmax sino la sigmoidea)

**LOG**: usamos log y no la prob de la clase solamente por varias cosas:

Cuando estamos entrenando, el modelo no mira de a una loss por ejemplo, sino lo entrenamos de a batches (varios ejemplos al mismo tiempo), entonces lo que queremos hacer es maximizar la probabilidad conjunta de predecir bien todos los ejemplos 
(MAXIMUM LIKELIHOOD).
Esto implica MULTIPLICAR muchas probabilidades… y cuando hacemos eso los numeros quedan muy muy chicos, entonces es preferible usar LOG de la prob y sumarlos, por las propiedades de los logaritmos. Es un truco. Y usamos negativo porque si no queda negativo.
Por eso se llama NEGATIVE LOG LIKELIHOOD.

$$
L = \prod_i p_i\\
logL = \sum_i log(p_i)\\
L_{NLL} = - \sum_i log(p_i)\\
L_{NLLBatch} = - \frac{1}{B}\sum_i log(p_i)
$$

El maximum likelihood y el negative log likelihood son lo mismo pero en practica se usa el log para hacer operaciones con numeros no tan chicos (que pueden tener errores importantes).

Log es diferenciable y smooth. Tambien porque tiene buenas propiedades. En vez de usar el (1-p) que es lineal, esto le da mucha mas fuerza exponencial a los errores grandes.

![image.png](AI%20notas/image%2040.png)

![image.png](AI%20notas/image%2041.png)

Clasificacion multiclase:

Se usa distribucion 

Cross entropy loss EN UNA RED SIMPLE LINEAL → con sigmoid al final solamente pero no otra activacion

![image.png](AI%20notas/image%2042.png)

![image.png](AI%20notas/image%2043.png)

**Cross Entropy sobre una sola sigmoide no genera mínimos locales**, porque la función es convexa respecto a los logits. Pero si agregás **no linealidades como tanh, ReLU, o varias capas**, empezás a obtener superficies más complejas.

Cross entropy con red con una no-linealidad simple (ReLU)

$$
yhat = σ(w_2 * ReLU(w_1 * w))
$$

![image.png](AI%20notas/image%2044.png)

---

## Gradient descent

Como ya vimos, el objetivo para ENTRENAR un modelo (fittearlo a la data para que aprenda y poder hacer predicciones), es **MINIMIZAR LA LOSS.**

Esto se hace de a pasos, de una forma numerica/computacional, con un algoritmo de aprendizaje. Esto se llama **algoritmo de optimizacion**.

Entonces, **todo machine learning se puede reducir a un problema de optimizacion, donde uno tiene datos, un modelo, una loss function y tiene que minimizar esa loss, ajustando los parametros del modelo.**
El algoritmo de optimizacion → depende del tipo de funcion

- Funciones diferenciables: suaves, continuas, sin saltos ni esquinas. Tiene pendiente en cada punto de la funcion.
- Funciones no diferenciables: Pueden tener discontinuidades, saltos, etc. Hay puntos donde no tienen una pendiente clara (escalon, absolute, etc).

Algunos algoritmos sirven para cualquier tipo, como los evolutivos (geneticos), random search, etc. En definitiva, los algoritmos de optimizacion son algoritmos de busqueda. Pueden ser mas tontos como random search o grid search o pueden ser mas inteligentes (tienen una estrategia para ir mas rapido, como heuristicas o basados en gradiente).

Optimizacion basada en el gradiente: Si tenemos una funcion diferenciable, aprovechamos que se puede calcular el gradiente para informarte para donde va la funcion si cambiamos sus parametros.

Si podés calcular derivadas de manera estable y barata → **usá gradient descent**. Si **no podés**, o la función es muy discontinua, ruidosa o rara → pensá en métodos evolutivos o aleatorios.

**Redes neuronales como composicion de funciones diferenciables**:

Se puede ver a la red neuronal como un grafo de operaciones basicas diferenciables. Por ejemplo las multiplicaciones de los inputs con los weights, las sumas de los bias, el ReLU (que no es diferenciable completamente pero casi y se puede estimar sin problemas), el softmax (son otras operaciones basicas diferenciables), el calculo de la loss, etc.

![image.png](AI%20notas/image%2045.png)

Operaciones elementales:

- **Multiplicación**: `w * x`
- **Suma**: `+ b`
- **Máximo**: `ReLU = max(0, z)`
- **Exponencial**: `exp()`, usada en `sigmoid`, `softmax`, `tanh`
- **Logaritmo**: `log()`, en la loss cross-entropy

Estas operaciones tienen **derivadas analíticas simples.**

| Operación | Fórmula | Derivada | Aparece en... | Notas |
| --- | --- | --- | --- | --- |
| Suma | z = a + b | dz/da = 1, dz/db = 1 | Todas partes | Lineal |
| Resta | z = a - b | dz/da = 1, dz/db = -1 | Todas partes | Lineal |
| Multiplicación | z = a * b | dz/da = b, dz/db = a | Pesos por inputs | Producto derivado |
| División | z = a / b | dz/da = 1/b, dz/db = -a / b^2 | Softmax, normalizaciones | b ≠ 0 |
| Negación | z = -a | dz/da = -1 | Activaciones, pesos |  |
| Exponencial | z = exp(a) | dz/da = exp(a) | Sigmoid, softmax, tanh | Siempre positiva |
| Logaritmo | z = log(a) | dz/da = 1 / a | Cross-entropy | Solo definido para a > 0 |
| Potencia | z = a^n | dz/da = n * a^(n-1) | MSE (con n = 2) |  |
| Máximo (ReLU) | z = max(a, b) | dz/da = 1 si a > b, 0 si a < b | ReLU | No derivable en a = b |
| Inverso | z = 1 / a | dz/da = -1 / a^2 | Sigmoid | a ≠ 0 |

Ejemplos:

SIGMOID:

$$
σ(x) = \frac{1}{1+e^{-x}}
$$

1. neg_x = -x → negación
2. exp_neg_x = exp(neg_x) → exponencial
3. denom = 1 + exp_neg_x → suma
4. result = 1 / denom → división (inversa)

SOFTMAX:

$$
softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

- Para cada elemento x_i → compute exp(x_i) → exponencial
- sum_all = suma de todos los exp(x_i) → suma
- Para cada i → softmax_i = exp(x_i) / sum_all → division

MSE:

$$
MSE = \frac{1}{N} \sum_i^N (y_i-yhat_i)^2
$$

- error = y_pred - y_true → resta
- squared_error = error * error → multiplicacion
- sum_squared = suma de todos los squared_error → suma
- mse = sum_squared / N (cantidad de ejemplos) → division

NEGATIVE LOG LIKELIHOOD:

$$
L_{NLL} = - \sum_i log(p_i)
$$

- log_pred = log(pred) → log
- log_pred_sum = suma de todos los logs → suma
- nll = - log_pred_sum → negacion

ACLARACION: 

Se podrian tener otras operaciones no diferenciables en una red, siempre que se cumpla:

- No tienen parametros aprendibles (tiene parametros fijos o seteados de otra manera): porque no podes calcular el gradiente para esos parametros. Por ej si tenes if o reglas, o umbral, etc. No hay nada que optimizar en esas operaciones.
- No tienen parametros aprendibles ANTES de ellas (no interrumpen el flujo del gradiente): si vos tenes operaciones diferenciables con parametros aprendibles y despues estas no diferenciables, la señal del gradiente nunca va a llegar a los parametros aprendibles esos.

**Una red neuronal no tiene que ser 100% diferenciable en todas sus partes**. Solo tiene que serlo **en el camino entre los pesos aprendibles y la función de pérdida**.

GRADIENT VECTOR: Cuando tenemos mas de un parametro (multi-variable), tomamos las derivadas parciales de la loss con respecto a cada uno de los parametros. Y de ahi se forma un vector que es el gradient vector. Obtenemos:

- N direcciones (direccion del vector)
- Magnitud del cambio para cada parametro/dimension

Si tenemos 2 variables, el gradiente es:

$$
\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
$$

Desde un punto de la funcion (un valor para cada variable), miramos para x y para y como cambia la funcion, manteniendo la otra variable fija.

- Si a medida que estamos avanzando en x (desde el punto inicial), la funcion va disminuyendo, el slope es negativo, por lo que la derivada parcial es negativa.
- Si aumentar un poquito x hace disminuir la funcion mucho, entonces el slope es negativo y bastante mas grande
- Si aumentar un poquito x hace aumentar la funcion muy poquito, entonces el slope es positivo pero no tan grande
- Si aumentar un poquito x hace aumentar la funcion bastante, entonces el slope es positivo y bastante mas grande
- Lo mismo para y

Entonces por ej, si tenemos slope de x = -0.1, slope de y = 0.02… el gradient vector es [-0.1 , 0.02].

Que nos dice esto? Nos indica cuanto y para donde crece la funcion. Si hacemos un plot con el vector como coordenadas, x es negativo y bastante y y es positivo poquito. Esto genera una flecha que te indica la direccion donde mas crece la funcion f.

**Apunta al lugar donde mas sube la funcion**: Cada coordenada del gradiente te dice cuan sensible es la funcion al cambiar x o y. Para que f crezca, debes moverte donde ambas variables cambian, proporcionalmente a su importancia.

![image.png](AI%20notas/image%2046.png)

Entonces, **al tomar el negativo del gradient vector, apuntamos al lugar que disminuye la funcion**.

**ALGORITMO de gradient descent**

Es un algoritmo para encontrar el punto más bajo de una función.

Imaginá que tenés una función f(w) que querés minimizar. Por ejemplo, algo como una montaña invertida o un bowl.

1. **Elegís un punto inicial** w (donde estás parado)
2. **Calculás la pendiente** en ese punto: eso te dice hacia dónde sube la función.
3. **Dás un paso en la dirección contraria** a la pendiente (porque querés bajar, no subir)
    
    w ← w - η * f'(w)     w: posicion actual, η: tamaño del paso, f'(w): pendiente en ese punto
    
4. **Repetís** hasta llegar a un punto donde la pendiente sea casi cero (un mínimo)

**Aplicado a redes neuronales (simple)**

Queremos que un modelo **prediga un valor `y_hat` a partir de un input `x`**, usando una función muy simple con parámetros ajustables.
Tenemos una funcion lineal simple:

```python
y_hat = w * x + b
```

- `w`: pendiente (peso)
- `b`: bias (intersección)
- `x`: input
- `y_hat`: predicción del modelo

Queremos ajustar `w` y `b` para que `y_hat` se parezca lo más posible a `y`, que es el **valor real**.

Loss (Esta es la **función que queremos minimizar** con gradient descent.)

```python
loss = (y - y_hat)^2
     = (y - (w * x + b))^2
```

Paso a paso:

1. **Elegís valores iniciales** para `w` y `b` (por ejemplo, 0 y 0)
2. **Calculás la predicción** `y_hat = w * x + b`
3. **Calculás el error** `loss = (y - y_hat)^2`
4. **Calculás el gradiente**:
    - Derivada parcial del loss respecto a `w`:
        
        ```
        ∂loss/∂w = -2 * x * (y - y_hat)
        ```
        
    - Derivada parcial del loss respecto a `b`:
        
        ```
        ∂loss/∂b = -2 * (y - y_hat)
        ```
        
5. **Actualizás los parámetros**:
    
    ```
    w ← w - η * ∂loss/∂w
    b ← b - η * ∂loss/∂b
    ```
    
    Donde `η` es el learning rate (ej: 0.01)
    
6. **Repetís** los pasos 2 a 5 muchas veces hasta que el error se reduzca.

Como calculamos la derivada parcial de la loss con respecto a w (o a b)?

```python
∂loss/∂w = -2 * x * (y - y_hat)
```

Lo descomponemos en sub-operaciones y derivamos:

1. `z = w * x` → ∂z/∂w = `x`
2. `y_hat = z + b` → ∂y_hat/∂z = `1`
3. `e = y - y_hat` → ∂e/∂y_hat = `-1`
4. `loss = e²` → ∂loss/∂e = `2 * e`

Usamos la regla de la cadena:

```python
∂loss/∂w = ∂loss/∂e · ∂e/∂y_hat · ∂y_hat/∂z · ∂z/∂w
```

Y juntamos todo:

```python
∂loss/∂w = (2 * e) * (-1) * (1) * (x)
          = -2 * e * x
          = -2 * (y - y_hat) * x
```

Bien, tenemos una funcion simple y podemos calcular el gradiente (lugar que hace que la funcion crezca mas) y si tomamos un paso en el sentido inverso, podemos acercarnos a donde la funcion (de loss) disminuye.

Que pasa si tenemos una red neuronal deep, con muchas capas… como hacemos para llegar a actualizar hasta los primeros parametros? Como hacemos para actualizar todo?

## Backpropagation

Backprop es la forma para actualizar todos los parametros del modelo.

Usamos la **regla de la cadena** para ir calculando las derivadas parciales de los parametros, empezando por los ultimos (de la loss) y llegando hasta los primeros (los mas cerca del input).

CODIGO DE DEMOSTRACION EN NOTEBOOK

Idea conceptual:

Si tenemos una simple red con:

$$
z1 = w1x+b1 \\
a1 = \sigma(z1) \\
z2 = w2a1+b2 \\
yhat = \sigma(z2)
$$

El error es:

$$
e = (y-yhat) \\
L = e^2
$$

Para ACTUALIZAR los dos conjuntos de weights y bias, tenemos la partial derivatives:

$$
\frac{\partial L}{\partial w2}, \frac{\partial L}{\partial b2}, \frac{\partial L}{\partial w1}, \frac{\partial L}{\partial b1}
$$

Usamos la regla de la cadena, por ej para w2:

$$
\frac{\partial L}{\partial w2} = \frac{\partial L}{\partial e} \frac{\partial e}{\partial yhat} \frac{\partial yhat}{\partial z2} \frac{\partial z2}{\partial w2}
$$

Cada uno de las partial derivatives se calcula facilmente:

$$
L = e^2 \Rightarrow 
\frac{\partial L}{\partial e} = 
\frac{\partial e^2}{\partial e} = 2e \\
e = y - yhat \Rightarrow
\frac{\partial e}{\partial yhat} = -1 \\
yhat = \sigma(z2) \Rightarrow
\frac{\partial yhat}{\partial z2} = yhat(1-yhat) \\
z2 = w2a1 + b2 \Rightarrow
\frac{\partial z2}{\partial w2} = a1 and \frac{\partial z2}{\partial b2} = 1  \\
$$

Entonces reemplazando y modificando, nos queda:

$$
\frac{\partial L}{\partial w2} = 2ea1(yhat(1-yhat))
$$

Todo esto es para actualizar un solo weight (w2), de la ultima capa… pero que pasa, 
YA TENEMOS TODOS LOS VALORES CALCULADOS (cuando hicimos el forward pass):

- a → activacion
- e → error individual
- yhat → output de la red

Los tenemos que guardar y listo. Para otras tambien hay que guardar z1, x, w2, etc…

No estás recalculando derivadas como en symbolic math, ni estás haciendo aproximaciones como en finite differences. Solo:

1. **Forward pass:** calculás y guardás valores intermedios.
2. **Backward pass:** aplicás derivadas locales usando la cadena.

Y para las capas anteriores? Para la primera capa, w1, b1…

Para w2 era asi:

$$
\frac{\partial L}{\partial w2} = \frac{\partial L}{\partial e} \frac{\partial e}{\partial yhat} \frac{\partial yhat}{\partial z2} \frac{\partial z2}{\partial w2}
$$

Y para w1 es asi:

$$
\frac{\partial L}{\partial w1} = \frac{\partial L}{\partial e} \frac{\partial e}{\partial yhat} \frac{\partial yhat}{\partial z2} \frac{\partial z2}{\partial a1} \frac{\partial a1}{\partial z1} \frac{\partial z1}{\partial w1}
$$

Como usamos la regla de la cadena, ya hasta tenemos guardado casi todas las partial derivatives que necesitamos hasta ese punto! 

Entonces solo calculamos las que nos falta (ya tenemos todos los valores necesarios del forward pass) y listo.

Ojo, tambien hay que guardar una COPIA de los weights originales en ese punto (antes de ser actualizados), por ej necesitamos w2 para el calculo del grad para w1. Y w2 ya lo actualizamos cuando llegamos a w1… asi qeu tenemos que tener una copia de todos los weights que afectan a otros weights anteriores.

En muchos frameworks (PyTorch incluido), **no hace falta guardar una copia de los pesos**, porque:

- El backward no los modifica.
- El update de pesos (con el gradiente) ocurre **después** de que todo el backward ya terminó.

ES SUPER PODEROSO, PORQUE PODEMOS OPTIMIZAR CUALQUIER FUNCION ARBITRARIA MIENTRAS SUS OPERACIONES SEAN DIFERENCIABLES. Se puede diseñar para REPRESENTAR lo que queramos.

Consideraciones computacionales y de implementacion:

pytorch y micrograd (para demo) crean un GRAFO de operaciones que es definida por la red neuronal. 

Cada valor y operacion (que genera un nuevo valor) se guarda en una clase (micrograd Value y pytorch tensor). Estos son los nodos del grafo, y cada nodo guarda ciertas cosas:

- .data: el valor real
- ._backward: la funcion para hacer backward pass desde ese nodo
- ._prev: son los children (todos los nodos anteriores, que produjeron este nuevo)
- ._op: el tipo de operacion
- .grad: el gradiente acumulado de ese nodo hasta ese momento (los grads que le va llegando)

Paso a paso: PyTorch maneja esto así (parecido micrograd pero con Value class):

- Durante `forward`:
    - Cada `Tensor` que participa en operaciones guarda:
        - Su valor (`.data`)
        - Su operación (`.grad_fn`)
        - Sus inputs (implícitos en `grad_fn`)
- Durante `backward()`:
    - Va caminando desde la loss hacia los inputs
    - Calcula `grad` en cada nodo. 
    El gradiente acumulado de ese nodo hasta ese momento (los grads que le va llegando)
    - No recalcula valores intermedios: **usa los guardados**

UN PARAMETRO PUEDE AFECTAR LA LOSS POR MULTIPLES CAMINOS:

Esto pasa cuando hay batches de datos… w1 influye en la loss por cada ejemplo.

Entonces cuando hacés `L.backward()`, la derivada de `L` respecto a `w` debe **sumar los efectos de todos los caminos:** 
Para simplificar, tenemos w que pasa por f1 y f2 y eso influye en L:

$$
\frac{\partial L}{\partial w} = 
\frac{\partial L}{\partial f1} \frac{\partial f1}{\partial w} + \frac{\partial L}{\partial f2} \frac{\partial f2}{\partial w}
$$

Se SUMAN las contribuciones. Pytorch lo hace automaticamente, recorriendo todo el grafo en orden topologico inverso. 

```python
# cuando llega a f1 hace
w.grad += ∂f1/∂w * grad_output

# cuando llega a f2 hace
w.grad += ∂f2/∂w * grad_output
```

Usa += siempre. Si no lo hiciera siempre se quedaria con el gra del ultimo camino.

Por eso siempre hay que RESETEAR LOS .grad antes de cada iteracion de entrenamiento:

```python
optimizer.zero_grad()
```

Con pytorch, lo interesante tambien es que se procesa todo el batch EN PARALELO. Entonces tenemos como input algo de 2D. La primera dimension es el ejemplo en el batch y la segunda son las caracteristicas / features / secuencia del ejemplo.

## Estabilizacion del entrenamiento

Problemas, optimizers, initialization, batches, epochs

Ya sabemos Teoricamente como es el entrenamiento del modelo: **gradient descent con backpropagation.** 
Pero en la practica aparecen problemas, por eso tenemos que **estabilizar** el entrenamiento, para que converja de manera estable rapida y eficiente (sin desperdiciar recursos) hacia un minimo razonable.

Tres grandes tipos de problemas:

- **Inestabilidad:** la loss sube y baja violentamente, no converge, oscila demasiado.
- **Entrenamiento lento:** la loss baja demasiado lento, necesitás muchas iteraciones o epochs.
- **Mínimos malos:** quedás atrapado en mínimos pobres (locales), lejos del mínimo óptimo que querías encontrar.

¿Por qué pasa esto?

- **Gradientes demasiado grandes o demasiado chicos (vanishing/exploding gradients)**
- **Mal inicializados los pesos**
- **Learning rate mal elegido**
- **Ruido en la estimación del gradiente (batch size demasiado chico o demasiado grande)**

Y otros problemas:

### Local minima / saddle point

**Quedarse en mínimo local**

- **Qué pasa:** el modelo se estanca en una solución mala.
- **Por qué pasa:**
    
    La superficie de la loss es **no convexa** (muchos valles y colinas).
    
    ![image.png](AI%20notas/image%2047.png)
    
    El gradiente puede llevarte a un **mínimo pobre** del cual no podés escapar.
    
- **Solución + Por qué funciona:**

| Solución | Por qué funciona |
| --- | --- |
| **Momentum** | Puede **acumular suficiente velocidad** como para salir de un valle superficial. |
| **Mini-batches** | Introducen “ruido” en el gradiente, lo cual puede ayudarte a “saltar” fuera de mínimos locales. |
| **Varios reinicios (random seeds)** | Entrenar desde distintos puntos iniciales da más chances de caer en mejores mínimos. |

Ya sabemos que la loss function puede ser muy compleja y tener muchos local minima y saddle points (puntos en el medio que la gradiente es 0)..

**Si empezamos desde una posicion aleatoria y hacemos gradient descent, nada nos asegura que terminemos en el global minimo**.

Podriamos hacer busqueda exhaustiva o empezar desde muchos lados aleatoriamente y ver cual llega a un minimo pero no es computacionalmente accesible para redes muy grandes.

**STOCHASTIC GRADIENT DESCENT (SGD)**

La estrategia sigue igual, pero en cada paso se le añade aleatoriedad para donde se mueve, por lo que en un paso puede moverse incluso para donde sube en vez de bajar. En promedio siempre va a ir para abajo.

![Captura de pantalla 2025-06-30 a la(s) 14.47.36.png](AI%20notas/Captura_de_pantalla_2025-06-30_a_la(s)_14.47.36.png)

Como le añadimos aleatoriedad? con:

**BATCHES AND EPOCHS**

Para cada iteracion, el algoritmo elige aleatoriamente un **subconjunto de los datos de entrenamiento y computamos el gradiente** para solamente esos ejemplos. → **MINIBATCH**

Los batches usualmente se samplean del dataset SIN REEMPLAZO. Va tomando muestras hasta que termina todo el dataset. Y ahi empieza de nuevo. Todo un paso por el dataset entero (mediante minibatches) es un → **EPOCH**

El minibatch puede ser de 1 ejemplo o de todo el dataset → full-batch (es igual a no hacer stochastic).

Se puede pensar como que SGD calcula el gradiente para DIFERENTES LOSS FUNCTIONS, una para cada iteracion. 
**La loss function depende del modelo y de la data**, entonces va a ser diferente para cada minibatch.

![Captura de pantalla 2025-06-30 a la(s) 14.55.05.png](AI%20notas/Captura_de_pantalla_2025-06-30_a_la(s)_14.55.05.png)

Aca la primera es la general. Las otras tres son tres minibatches que generan una loss function diferente… entonces si vemos la general, es como que toma en cuenta todos estos distintos loss functions. Para cada minibatch, cambian los parametros del modelo para parecerse mas a la data (3 ejemplos) de ese minibatch.

$$
φ_{t+1} \longleftarrow φ_{t}-α \sum_{i∈B_t} \frac{\partial l_i(φ_t)}{\partial φ}
$$

O sea, sumamos (o promediamos) las losses individuales de cada ejemplo del batch y eso lo multiplicamos por el learning rate y con eso actualizamos.

Propiedades de SGD:

- Es mas barato computacionalmente calcular el gradiente para un subset de la data. O sea no tenes que esperar a computar la loss para todo el dataset sino que lo vas actualizando de apoco haciendo backwards mas seguido. Entonces no usas mucha memoria ni tiempo de computo para actualizar. 
Esto sirve tambien para hacerlo en GPUs.
- Podria ayudar a escapar local minima y saddle points
- Funciona mejor empiricamente para encontrar parametros que generalicen mejor

No necesariamente va a converger. Para mejorar esto y que no siga saltando de un lugar al otro (por la aleatoriedad de los batches), se le agrega un → **LEARNING RATE SCHEDULE.**

Empieza con el learning rate grande asi “explora” mas todo el territorio, y despues de varias epochs se va achicando asi ayuda a converger.

Otros problemas:

- En regiones donde la superficie de pérdida tiene forma de "valle alargado", puede oscilar mucho de un lado a otro (como una pelota rebotando en zig-zag).
- Si el gradiente cambia de dirección constantemente (ej. porque el minibatch cambia), la red da pasos que se anulan o que no avanzan en la dirección correcta.

**Momentum**

Se le suele agregar momentum. Cuando actualizamos los gradientes le agregamos un termino basado en la direccion en la que se movio en el paso anterior (pesado por un parametro beta).

$$
\begin{align*}m_{t+1} &\leftarrow \beta \cdot m_t + (1 - \beta) \cdot \sum_{i∈B_t} \frac{\partial l_i(φ_t)}{\partial φ} \\\phi_{t+1} &\leftarrow \phi_t - \alpha \cdot m_{t+1}\end{align*}
$$

**m** es un vector que va a ser igual al gradiente pero va a ser una combinacion (weighted suma) de las partial derivatives de cada parametro en los pasos anteriores hasta el presente: 

Inicializas m_0 (vector) = todos 0. Para cada paso:

- Elegis un minibatch (por ej 3 ejemplos):
    - Calculas la loss total del batch (suma o mean de las individuales)
    - Calculas el gradiente total del batch (¿Cómo cambiarían los parámetros del modelo (todos los parametros) si quisiera que la pérdida del batch baje?
    - Actualizamos el momentum, como es 0 queda el gradiente recien calculado
    
    $$
    m_{t+1}=β⋅m_t+(1−β)⋅\sum_{i∈B_t} \frac{\partial l_i(φ_t)}{\partial φ}
    $$
    
    - Actualizamos los pesos del modelo
    
    $$
    \phi_{t+1} \leftarrow \phi_t - \alpha \cdot m_{t+1}
    $$
    
- Elegis otro minibatch:
    - … lo mismos primeros 2 pasos
    - Para actualizar el momentum, ahora agarramos el momentum anterior, que es el primer gradiente calculado y SE LO SUMAMOS (weighted por beta) al gradiente nuevo calculado
    - …

Como tenemos esa SUMA PESADA por beta:

- Si en los pasos anteriores los gradientes estan alineados (siempre positivos o siempre negativos), el learning rate efectivo aumenta
- Si la direccion del gradiente va cambiando repetidamente, el learning rate efectivo disminuye porque se cancelan con las sumas

Cada gradiente anterior va quedando en el “historial” del momentum, pero con **peso cada vez menor**.
Es como una memoria que se va olvidando de a poco.

Aclaracion: **el gradiente de una suma es la suma de los gradientes.** Esto se cumple porque la derivada es una operación **lineal**:

$$
\sum_{i \in B_t} \frac{\partial \ell_i(\phi_t)}{\partial \phi} = \frac{\partial}{\partial \phi} \left( \sum_{i \in B_t} \ell_i(\phi_t) \right)
$$

![Captura de pantalla 2025-06-30 a la(s) 16.08.06.png](AI%20notas/Captura_de_pantalla_2025-06-30_a_la(s)_16.08.06.png)

Hasta el momentum, solo con GD, se calcula el gradiente (vector de partial derivatives para cada parametro del modelo) y para cada parametro tomamos esa partial derivative, lo multiplicamos por el learning rate y lo actualizamos.

Con momentum, m es un vector de misma dimension que el gradiente. **Para cada parámetro individual**, estás usando un resumen de cómo venía cambiando su gradiente en pasos anteriores.

Sin embargo, **los distintos parametros pueden tener gradientes muy distintos**.

- Capas más cercanas a la salida suelen tener gradientes más grandes.
- Algunas neuronas están más activas o conectadas con features más importantes.
- Otras están medio dormidas (muertas, saturadas, etc.), entonces su gradiente es casi cero.

Entonces si los gradientes de los parametros son muy distintos, **no es buena idea usar siempre el mismo learning rate.** Por ej cuando el gradiente de la loss es mucho mas steep en una direccion que en otra, es jodido elegir un learning rate.

Ejemplo:

Supongamos que tenés 3 parámetros: phi_t = [1.0, -0.5, 2.3]

Y el gradiente en ese paso es: g_t = [10.0, 0.01, -3.0]

Y usás un learning rate fijo alpha = 0.1, entonces:

$$
\phi_{t+1} = \phi_t - 0.1 \cdot g_t = [0.0, -0.501, 2.6]
$$

- El primer parámetro saltó de **1.0 a 0.0**. Saltazo.
- El segundo casi ni cambió. **Se estanca.**
- El tercero hizo un cambio razonable.

Queremos que cada parametro del vector tenga su propio paso, **ajustado dinámicamente**:

- Si el gradiente de una parametro es **grande todo el tiempo**, le bajamos el paso (porque ya sabemos que ahí el terreno es empinado).
- Si el gradiente es **pequeño**, le subimos el paso (porque necesitamos ayudarlo).

**Adam Optimizer**

Adam dice: *"Voy a ver cómo se vienen comportando esas derivadas, y les voy a ajustar el paso para que todos avancen a un ritmo más balanceado."*

- **Divide cada coordenada por su historial de magnitudes**.
- Así, incluso si un gradiente es grande, lo "frena".
- Y si es chico pero constante, lo "ayuda".

Una idea basica seria ESTANDARIZAR los gradientes (por su magnitud pasada) y que solo quede el signo (para donde se mueve) y que se muevan todos los parametros la misma cantidad. Se lleva un registro de los cuadrados de los gradientes pasados (dice que tan grandes son los gradientes en promedio para cada parametro).

Pero eso nunca va a converger porque va a seguir moviendose, asi que le podemos agregar momentum.

Adam → Adaptive Moment Estimation. Mezcla **normalizacion + momentum.**

---

### Problemas de activaciones y gradientes

Cuando entrenás redes profundas, los **valores** que fluyen hacia adelante (activaciones) y hacia atrás (gradientes) **se van deformando capa a capa**. Esto puede generar varios problemas graves:

Problemas comunes:

1. 🚨 **Exploding Gradients / Exploding Activations**

**Qué pasa:**

- Los valores (activaciones o gradientes) se vuelven **enormes** a medida que pasan por muchas capas.
- Las salidas se hacen inestables, aparecen `NaN`, `inf`, o el loss explota.
- El modelo no aprende o aprende mal y de forma errática.

**Por qué pasa:**

- Cada capa multiplica por una matriz (pesos) y aplica una activación.
- Si los pesos o activaciones **amplifican** aunque sea un poquito, al encadenar muchas capas eso **se acumula exponencialmente**.
- Especialmente en redes profundas, un pequeño error se convierte en un descontrol.

**Ejemplo intuitivo:**

- Imaginate que cada capa multiplica por 1.1 → después de 20 capas es 1.1^20 ≈ 7.
- Pero si es 1.5 por capa → 1.5^20 ≈ 3325 → ya está todo roto.

---

2. 🧊 **Vanishing Gradients**

**Qué pasa:**

- Las señales de error (gradientes) que se propagan hacia atrás se **hacen cada vez más pequeñas**.
- Las capas más cercanas a la entrada **no reciben señal útil para aprender**.
- El modelo se estanca: entrena solo las capas finales, las otras no cambian.

**Por qué pasa:**

- Pasa lo mismo que con exploding gradients, pero al revés: cada capa **achica** un poco la señal.
- Si multiplicás muchas veces por valores < 1 (por ej. 0.9), el gradiente se **aplana** hasta desaparecer.

**Ejemplo:**

- 0.9^10 ≈ 0.35 → todavía zafa.
- 0.5^10 ≈ 0.001 → ya no queda nada.

---

3. 🧱 **Saturación de activaciones (`tanh`, `sigmoid`)**

**Qué pasa:**

- La activación se queda trabada en los extremos del rango (cerca de 0 o ±1).
- En esas zonas, **la derivada es casi 0** → la neurona deja de aprender.

**Por qué pasa:**

- Funciones como `sigmoid` o `tanh` son "squashing": llevan todo al intervalo [0,1] o [−1,1].
- Pero **su derivada depende de estar en la parte central de la curva**.
- Si `z = Wx + b` es muy grande o muy chico, cae en la cola de la función → derivada ≈ 0.

**Consecuencia:**

- Aunque el gradiente global diga “hay que cambiar algo acá”, **no llega señal** a esa neurona.
- La red **no puede corregirla** → queda atrapada.

---

4. ⚰️ **Neuronas muertas (`ReLU`)**

**Qué pasa:**

- Una neurona empieza a dar siempre 0 y nunca más se recupera.
- Queda completamente desconectada del aprendizaje.

**Por qué pasa:**

- `ReLU(z) = max(0, z)` → para `z < 0`, la salida es 0.
- La derivada de ReLU en `z < 0` es 0 → no hay gradiente → no se puede ajustar.
- Si una neurona **recibe siempre `z < 0`**, se muere.

**Cómo puede pasar eso:**

- Mala inicialización (muchos pesos negativos)
- Inputs mal escalados
- Learning rate muy alto que empuja los pesos a regiones negativas permanentes

---

5. 🎲 **Logits iniciales desbalanceados (en clasificación)**

**Qué pasa:**

- Al comenzar el entrenamiento, la red produce logits que favorecen una clase al azar.
- El `softmax` da algo como `[0.95, 0.02, 0.01, 0.01, ...]` sin razón real.
- La loss es altísima, y los primeros pasos se pierden aplastando esa locura.

**Por qué pasa:**

- Al inicializar, los pesos son aleatorios → la salida también.
- Pero como el `softmax` **responde a diferencias relativas**, aunque los logits sean “ruido”, puede generar distribuciones muy sesgadas.
- Eso genera **grandes errores iniciales** y hace que **el entrenamiento arranque mal**.

Estos problemas afectan **tanto al forward pass como al backward pass**.

Se van acumulando capa a capa, y si no los controlás, el entrenamiento se rompe. Todos tienen una **misma raíz**: los valores (`z`, activaciones, gradientes) **se deforman capa a capa**, porque cada capa transforma la información.

Approaches para solucionarlos:

- La **inicialización** previene que el modelo arranque mal (explosiones o colapsos al inicio)
- La **estabilización forzosa** mantiene el entrenamiento sano a lo largo del tiempo

Ambas cosas **apuntan a lo mismo**: que las señales (activaciones y gradientes) se mantengan **en un rango útil y saludable**.

Si el modelo aprende es porque la información circula. Si la información se aplasta o explota, el aprendizaje se rompe.

---

### **Inicializacion**

**Inicializacion de loss (logits iniciales desbalanceados)**

- **Qué pasa:** la red empieza en un estado confuso, con una distribucion de logits muy desbalanceada y tiene que aplastar los logits y desperdicia los primeros pasos del entrenamiento hasta **aplanar** esa distribucion.
- **Por qué pasa:**
    
    Cuando recién inicializás una red neuronal para clasificación, la **última capa** suele producir un vector de valores (logits) que luego pasan por un `softmax` para convertirlos en una **distribución de probabilidad sobre las clases**.
    
    El `softmax` es sensible a **los valores relativos de los logits**:
    
    - Si todos los logits son iguales → `softmax` da una **distribución uniforme**.
    - Si uno es mucho mayor que los demás → `softmax` da **una clase dominante con alta probabilidad** (baja entropía).
    
    Cuando inicializás con pesos aleatorios, **no hay ningún control** sobre:
    
    - la **magnitud** de los logits
    - ni su **distribución relativa**
    
    Entonces, puede pasar que:
    
    - El softmax inicial sea **muy desbalanceado** (p. ej., una clase con 95% de probabilidad)
    - El `negative log likelihood` sea **muy alto**
    - La red empiece en un estado **confuso**, donde tiene que primero "aplastar" esa salida loca, y **desperdicia los primeros pasos del entrenamiento corrigiendo eso**
- **Solución + Por qué funciona:**

| Solución | Por qué funciona |
| --- | --- |
| **Subir la entropia en logits inicial** | Queremos que la primera salida de logits sea mas o menos uniforme, asi no tarda tiempo corrigiendo o sacandole ruido a la distribucion. Entonces: distribucion mas o menos uniforme y logits cercanos a 0. Asi no favorece a ninguna clase y hace que la loss inicial no sea muy alta ni haya un bias inicial alto.
Para esto, forzamos los parametros de la ultima capa (output) a que sean casi todo 0:
- b = 0 → no favorece a ninguna clase
- W casi 0 pero le dejamos un poco de entropia → hace que X@W de valores pequeños.
Por que W ≠ 0? Porque el gradiente que llega seria el mismo para todas las clases, el modelo queda atrapado haciendo siempre lo mismo. Necesitamos asimetria. |

HAY CODIGO EN LA NOTEBOOK

---

**Pesos iniciales mal elegidos (todos iguales o cero)**

- **Qué pasa:** el modelo no aprende o aprende lo mismo en todas las neuronas.
- **Por qué pasa:**
    
    Si inicializás todos los pesos igual, todas las neuronas hacen lo mismo → sus gradientes son idénticos → se actualizan igual → **no hay aprendizaje diferencial**. Se pierde la simetría.
    
- **Solución + Por qué funciona:**

| Solución | Por qué funciona |
| --- | --- |
| **Inicialización aleatoria** | Introduce diferencias entre neuronas desde el inicio → cada una aprende de forma distinta. Esto rompe la simetría y permite especialización. |

---

**Varianza de los parametros en inicialización**

Cuando inicializás una red neuronal, los **pesos aleatorios** pueden causar dos problemas graves:

1. ❌ **Vanishing gradients**: los gradientes se hacen muy chiquitos, no llega señal para aprender.
2. ❌ **Exploding gradients**: los gradientes se hacen enormes y explotan.

Estos problemas aparecen porque a medida que **las señales se propagan hacia adelante (activaciones)** y hacia atrás (gradientes), se **amplifican o se achican demasiado**. Si esto pasa por muchas capas, se rompe todo.

Lo que **queremos** es que:

- Las pre**activaciones** (`z = W @ x`) tengan más o menos **media 0** y **varianza 1**
- Y que los **gradientes** también se mantengan estables durante el backpropagation

→ esta varianza es TEORICA, no es la varianza entre neuronas! Es asumiendo que X y W son variables aleatorias.

La varianza teorica depende de:

- Los **inputs** (que vienen de la capa anterior)
- Los **pesos** que multiplicamos

Entonces:

Si los **inputs** tienen varianza 1, y queremos que el **output también tenga varianza 1**,

→ tenemos que elegir los **pesos con un cierto tamaño (std)**.

Como z (el output o pre-activacion) cuando x y w vienen de N(0,1) es:

$$
z=∑_{i=1}^nw_i⋅x_i
$$

Va a tener: 

- mean = 0
- **Varianza = n = fan_in.**

fan_in = n → es la cantidad de elementos que suma la neurona para hacer z (la cantidad de neuronas en la layer anterior o input). 

Si tenemos 3 inputs a la neurona, la varianza va a ser de 3. O sea que si el input es grande, la salida (los ouputs de las neuronas de la capa) van a ser muy ruidosos.

Si z es muy grande → las activaciones se saturan cuando tienen squashing function y el grad se hace 0.

**Escalar pesos por fan-in**:

Para que z tenga varianza = 1, los pesos deben tener varianza = 1/n. 

$$
w_i∼N(0,\frac{1}{n})
$$

Entonces iniciando los parametros con esa distribucion, deberia normalizar y estabilizar la distribucion de las preactivaciones.

**Gain**

Es un factor que se le agrega multiplicando la varianza para crear parametros estables → **depende de la funcion de activacion.**

- Para **ReLU**: gain = √2 → te queda lo de He. Esto es porque ReLU descarta la mitad de los valores (pone en cero los negativos), entonces para compensar, duplicamos la varianza esperada.

$$
w_i∼N(0,\frac{2}{fan_{in}})
$$

- Para `tanh`: gain = 5/3 → más chico porque `tanh` achica los valores
- Para `linear`: gain = 1 → te queda Xavier normal

Generalizando:

$$
w_i \sim \mathcal{N}\left(0, \frac{\text{gain}^2}{\text{fan\_in}} \right)

$$

![Captura de pantalla 2025-06-30 a la(s) 18.39.44.png](AI%20notas/Captura_de_pantalla_2025-06-30_a_la(s)_18.39.44.png)

Y si la cantidad de inputs y de outputs es distinta? Por ej con capas con distinta cantidad de neuronas (fan_in y fan_out).

Cuando inicializás una red, vos querés que:

- Las **activaciones** (forward pass) tengan una varianza razonable (≈1)
- Los **gradientes** (backward pass) también tengan una varianza razonable (≈1)

Pero hay un problema cuando:

> La matriz de pesos no es cuadrada, es decir, la cantidad de neuronas en la capa anterior (D_h) y en la capa siguiente (D_h′) son diferentes.
> 

La propagación hacia adelante y hacia atrás **se ven afectadas de distinta forma** por el tamaño de las capas:

- En el **forward pass**, la varianza de las salidas (`z = Wx`) depende del número de entradas → `fan_in = D_h`
- En el **backward pass**, la varianza de los **gradientes** depende del número de salidas → `fan_out = D_h′`

Entonces:

- Si inicializás usando solo `fan_in`, estabilizás el forward, pero el backward puede romperse.
- Si usás `fan_out`, estabilizás el backward, pero el forward puede romperse.

> ❗ No podés satisfacer ambas condiciones al mismo tiempo si la matriz de pesos no es cuadrada.
> 

Entonces, se propone un **compromiso** entre los dos:

$$
\sigma^2_{\Omega} = \frac{4}{D_h + D_h'}

$$

Esto es:

- En lugar de usar solo `fan_in` o `fan_out`, usamos el **promedio** (`(D_h + D_h′)/2`)
- Y la constante `4` sale de combinar ambas fórmulas (una para forward y otra para backward)

**CRITICAS**:

**Qué pasa si `x` (los inputs) no tienen varianza 1?**

Todas las fórmulas de inicialización (como He, Xavier) suponen que los inputs x tienen media 0 y varianza 1.

Pero en la práctica, eso no siempre pasa:

- Algunas **features pueden tener valores más grandes** que otras
- Algunas pueden tener **media ≠ 0**
- Y puede haber **diferencias grandes entre features**, lo que rompe la suposición

Esto hace que las **preactivaciones `z = Wx` salgan desbalanceadas**, y todo lo que planeamos para que tenga varianza ≈ 1 ya **no se cumple**.

Cómo se soluciona?

- Usamos técnicas como **BatchNorm** o **LayerNorm** para **corregir dinámicamente las estadísticas del input** en cada forward pass
- Así, incluso si `x` no está bien distribuido, **podemos seguir entrenando sin romper la red**

**¿Y si entrenamos muchas épocas… se rompe igual?**

Aunque arranquemos con buenas varianzas, después de muchos updates los pesos pueden desviarse, crecer mucho o achicarse

Esto pasa porque:

- El entrenamiento **ajusta los pesos**
- Las activaciones y los gradientes **se propagan capa a capa**
- Si no cuidamos eso, **la varianza se puede ir acumulando o reduciendo** en cada paso

Esto rompe lo mismo que queríamos evitar al inicio:

- Las activaciones se saturan (vanishing gradients)
- O se disparan (exploding gradients)
- O **las neuronas se “mueren”** (por ejemplo con ReLU si quedan en región negativa para siempre)

Cómo se soluciona?

- Usamos herramientas de **estabilización durante el entrenamiento**:

| Técnica | Para qué sirve |
| --- | --- |
| **BatchNorm / LayerNorm** | Corrige las estadísticas de las activaciones automáticamente |
| **Gradient clipping** | Limita la magnitud de los gradientes para evitar explosiones |
| **Optimizers como Adam** | Adaptan los updates para evitar desbalances |
| **Learning rate schedules** | Evitan saltos bruscos al final del entrenamiento |
| **Weight decay** | Evita que los pesos crezcan demasiado |

---

La **inicialización** te da un **buen punto de partida**.

Pero durante el entrenamiento, necesitás **mecanismos para mantener la estabilidad**.

> La inicialización previene problemas al principio.
> 
> 
> La normalización y otros métodos los previenen mientras entrenás.
> 

---

### Estabilizacion forzosa (Norm)

Por que queremos activaciones estables? Segun como sean los parametros, las activaciones pueden ser muy chicas o muy grandes y esto puede que en el forward pass, cada vez se hagan mas chicas o mas grandes… entonces esta bueno tener una estabilidad para que los valores no se saturen ni se mueran 
→ **vanishing gradient problem o exploding gradient problem**

**Saturacion de activaciones (tanh, sigmoid)**

- **Qué pasa:** muchas neuronas quedan atrapadas en la zona “flat” de la activación (≈ -1 o ≈ 1), lo que hace que **el gradiente que les llega sea casi cero**, y por lo tanto **no puedan aprender ni salir de ahí**. Se llaman *neuronas muertas* o saturadas.
- **Por qué pasa:**
    
    Las funciones como `tanh` o `sigmoid` son funciones “squashing”: toman cualquier valor real y lo reducen a un rango fijo:
    
    - `tanh(z)` ∈ [−1, 1]
    - `sigmoid(z)` ∈ [0, 1]
    
    El problema es que la derivada de estas funciones **se aplana** en los extremos:
    
    - `d/dz tanh(z) = 1 - tanh²(z)`
    - Cuando `tanh(z)` ≈ ±1, la derivada ≈ 0
    
    Esto implica que cuando `z = Wx + b` tiene valores muy grandes (en valor absoluto), entonces:
    
    - `tanh(z)` queda ≈ ±1
    - `dL/dz` ≈ 0
    - No hay gradiente → **no hay aprendizaje**
    
    Este efecto puede surgir:
    
    - Por **inicialización** con pesos grandes
    - Por inputs no normalizados
    - Por learning rate alto que empuja las activaciones fuera de rango
- **Solución + Por qué funciona:**

| Solución | Por qué funciona |
| --- | --- |
| **Inicialización Xavier (Glorot)** | Está diseñada para que la varianza de las activaciones **no crezca ni se reduzca** a medida que pasan por capas. Así evita que `z` sea demasiado grande o pequeño, manteniéndolo en la zona activa de `tanh` (≈ −2 a 2). |
| **BatchNorm / LayerNorm** | Normaliza las activaciones intermedias a media ≈ 0 y std ≈ 1, asegurando que la mayoría de los valores de `z` estén **en la región útil** de la función no lineal. Esto **previene que se saturen**. |
| **Usar ReLU o GELU** | ReLU no satura para valores positivos: `ReLU(z) = z` si `z > 0`, con derivada constante. Por lo tanto, **no tiene cola flat** como `tanh` o `sigmoid` (aunque puede dar neuronas muertas si `z < 0` siempre). |
| **Normalizar los inputs** | Si los inputs a la red tienen una varianza enorme o están descentrados, entonces `Wx + b` también lo está. Al normalizar, **mantenés los valores de entrada a las capas en un rango razonable**. |
| **Learning rate controlado** | Si el LR es muy alto, podés llevar los pesos a un régimen donde `Wx + b` se dispara → saturación. Usar un LR más bajo o con scheduler **previene que se salgan de rango durante el entrenamiento**. |

---

**Dead neurons (`ReLU`)**

- **Qué pasa:** algunas neuronas con `ReLU(x)` quedan permanentemente en cero, para todos los ejemplos. Como la derivada de ReLU para `x < 0` es 0, la neurona **nunca más se actualiza**.
- **Por qué pasa:**
    
    ReLU es:
    
    ```
    ReLU(x) = max(0, x)
    ```
    
    - Si `x < 0` → `ReLU(x) = 0`
    - Su derivada en esa zona es 0
    
    Entonces si por alguna razón el input a esa neurona está siempre en `x < 0`, esa neurona:
    
    - Da siempre 0
    - Tiene derivada 0
    - Nunca actualiza su peso
    - **Está muerta**
    
    Esto puede pasar por:
    
    - Mala inicialización (muchos pesos negativos)
    - Mal input (mal escalado o sin normalizar)
    - Learning rate muy alto que empuja todo el peso hacia negativo
- **Solución + Por qué funciona:**

| Solución | Por qué funciona |
| --- | --- |
| **Inicialización He (para ReLU)** | Inicializa los pesos con una varianza mayor, adaptada a la activación ReLU: `Var(W) = 2 / fan_in`. Esto **aumenta la probabilidad de que `x > 0` al menos algunas veces** → la neurona se activa. |
| **Leaky ReLU / GELU / ELU** | Variantes de ReLU que **no tienen derivada cero para `x < 0`**. Por ejemplo: `LeakyReLU(x) = 0.01x` si `x < 0`. Así, incluso si la neurona está “apagada”, **recibe algo de gradiente y puede revivir**. |
| **BatchNorm antes del ReLU** | Centra las preactivaciones para que `x ≈ 0 ± 1`. Esto **evita que todas las activaciones estén solo en la parte negativa** y ayuda a mantener neuronas vivas. |
| **Scheduler o warmup del learning rate** | Un LR muy alto puede empujar rápidamente los pesos a un lugar donde la neurona queda atrapada. Usar un warmup **da tiempo a que la red se acomode** antes de hacer pasos más grandes. |

---

**Explosión de activaciones**

- **Qué pasa:** los valores de activación intermedia crecen capa a capa, se vuelven enormes, y eso rompe el entrenamiento (produce `NaN`, `inf`, o gradientes enormes).
- **Por qué pasa:**
    
    A medida que una red profunda va pasando los outputs por más capas, cada una con multiplicaciones (`Wx`) y activaciones, la **varianza de las activaciones puede crecer acumulativamente** si no se controla.
    
    Especialmente si no tenés:
    
    - Inicialización que controle esa varianza
    - Normalización en el camino
    - Arquitectura que mitigue acumulación
- **Solución + Por qué funciona:**

| Solución | Por qué funciona |
| --- | --- |
| **Inicialización He o Xavier** | Estas inicializaciones **mantienen la varianza estable** entre capas, evitando que se acumule hacia arriba o hacia abajo. |
| **BatchNorm / LayerNorm** | Aseguran que la salida de cada capa esté en un rango manejable (media 0, std 1), **independientemente del comportamiento acumulado anterior**. |
| **Residual connections (como en ResNet)** | Permiten que los datos fluyan directamente sin pasar por tantas capas de transformación. **Evitan acumulación de transformaciones deformantes**. |

**BATCH NORMALIZACION** 

En lugar de confiar solo en la inicialización y esperar que todo funcione bien después, lo que podemos hacer es **forzar directamente que los outputs de las neuronas tengan una distribución estable y razonable**, incluso mientras la red está aprendiendo.

La idea es: **controlar la distribución de las preactivaciones**, en vez de esperar pasivamente que salgan bien.

Entonces, normalizamos la salida de cada capa, justo después de calcular: z = W x + b

**SE NORMALIZA CADA NEURONA POR SEPARADO, usando el mean y desvio para ESA MISMA NEURONA, a traves del batch.** No se usa informacion de otras neuronas sino de la misma pero en los distintos ejemplos del batch.

Para cada minibatch:

- se calcula la **media** y el **desvío estándar** de esos `z` (del minibatch)
- se normalizan:

$$
\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Pero si solo hiciéramos eso, la red estaría muy limitada. Por eso, después de normalizar le damos libertad para aprender una transformación:

$$
y_i = \gamma \cdot \hat{z}_i + \beta
$$

donde $\gamma$ y $\beta$ son parámetros **aprendibles**:

- $\gamma$ → escala (puede subir o bajar la varianza)
- $\beta$ → desplazamiento (puede mover el centro)

→ O sea: **la red puede decidir si quiere que las preactivaciones tengan más o menos varianza**, pero ahora lo hace **desde una base estable y controlada**.

- Sí, puede cambiar la distribución, **pero no desde cualquier caos**, sino desde un estado *bien comportado*.
- La red **aprende a deformar esa distribución solo si realmente lo necesita**, no por accidente.

**Por que se llama BatchNorm?**

Porque se usan las estadísticas **del batch actual**: 
La **media** y **desvío estándar** se computan sobre todos los ejemplos del batch para **cada neurona**.

En **inferencia** no tenemos batch. 

Lo que se hace es usar una media y varianza GLOBAL que se fueron acumulando durante el entrenamiento con **running average**. Despues lo guardamos y usamos eso al momento de inferencia para hacer las normalizaciones.

---

**LAYER NORMALIZATION**

En **BatchNorm** usamos las estadísticas del **batch** (media y varianza por neurona), pero eso **no sirve siempre**.

Por ejemplo:

- Si tu batch size es chico (o batch size = 1)
- O en **modelos autoregresivos o transformers**, donde procesás **una muestra a la vez**

BatchNorm **rompe** porque no tenés suficientes ejemplos para calcular bien las estadísticas.

> En vez de normalizar por neurona a lo largo del batch...
> 
> 
> **normalizamos por toda la capa, pero dentro de un único ejemplo.**
> 

Para cada ejemplo (no batch):

1. Agarras las pre-activaciones de una layer z
2. Calculas la media dentro del ejemplo (usando las pre-activaciones de la layer)
3. Calculas la varianza de eso
4. Normalizas cada componente del vector 

Tambien le damos la libertad con gamma y beta para que se mueva la distribucion.

Tambien se aplica antes de la activacion, despues de la pre-activacion.

```jsx
z = Wx + b
z_norm = LayerNorm(z) # ACA SE APLICA
h = activation(z_norm)
```

---

## Evaluacion (bias-variance)

Overfitting, generalizacion

Si dejamos crecer una red profunda y muy expresiva lo maximo posible, va a overfittear 

## Regularizacion

La idea es generalizar lo mejor posible.

**Regularizacion explicita**:

Es cuando le ponemos algun termino a la loss, para penalizar si pasan ciertas cosas.

Lo que le agregamos a la loss puede ser una funcion que tome cada parametro y segun el valor que tengan, los penalice… u otras cosas. Y le ponemos un peso a la influencia de esa penalizacion la ponemos con otro parametro.

$$
\text{LossReg} = \text{Loss} + \lambda \cdot g(\text{params})
$$

- lambda es el influenciador
- g es una funcion que toma los parametros y penaliza (con un escalar) lo que no le gusta

![image.png](AI%20notas/image%2048.png)

En este caso, el regularizador te da un mejor valor si los valores de los params 0 y 1 son proximos al centro y te penaliza si se alejan del centro. Al sumarlos queda asi.

**Se puede ver al regularizador como un PRIOR que conocemos (conocimiento sobre los parametros), para usar con el maximum likelihood criterion**.

L2 regularization (**weight decay** / ridge regression)

Penaliza la **suma de los cuadrados de los parametros**.

$$
\hat{\phi} = \arg\min_{\phi} \left[ \sum_{i=1}^{l} \ell_i[\mathbf{x}_i, \mathbf{y}_i] + \lambda \sum_j \phi_j^2 \right]
$$

Lo que hace es tratar de que la funcion se haga mas smooth, teniendo valores mas chicos en los parametros para que no se desvien mucho. 

Un efecto bueno es que si el modelo esta over parametrized, entonces en los lugares donde no hay datos, el modelo tiende a ser mas conservador en vez de hacer cualquier cosa.

**Regularizacion implicita** (por propiedades de gradient descent)

Gradient descent tiende a favorecer a algunos resultados antes que a otros.

Stochastic gradient descent → hace que mini batches pequeños funcionen mejor para generalizar por ciertas propiedades de implicit regularization.

**Metodos regularizadores**

Early stopping:

Frenar el entrenamiento despues de N steps, antes de que converja. Ayuda a reducir el overfitting porque si los parametros empiezan chiquitos desde casi cero, estas impidiendo que cambien mucho, entonces seria como L2. Se usa un hiperparametro.

Ensembling:

Construir muchos modelos y promediar las predicciones. La suposicion es que los errores de los modelos son bastante independientes por lo tanto deberian cancelarse un poco.

- La variabilidad de los distintos modelos se lo das inicializandolos aleatoriamente.
- Otro approach es bagging (bootstrap aggregating): se crean multiples datasets sampleando con replacement y entrenando los modelos con distintos datasets. Cuando un dato no esta en un modelo, lo va a tener que interpolarlo con los puntos cercanos, entonces asi puede ir armando algo mas smooth.
- Tambien otro approach es entrenar modelos con distintos hiperparametros.

Dropout:

Durante el entrenamiento, en cada iteracion de gradient descent, el modelo cancela aleatoriamente un subconjunto (alrededor de 50%) de hidden units (las deja en cero). Hace que la red no sea tan dependiente de algunas neuronas y que todas contribuyan un poco.

En inferencia, el modelo se ejecuta completo con todas las unidades, pero como ahora el modelo tiene mas unidades que con las que fue entrenada (en cualquier iteracion) multiplicamos los pesos por 1-dropout_probability para compensar (weight scaling inference rule).

Se rompe la cadena: en dropout, cuando se hace el forward pass, algnas hidden units se cancelan, quedan en cero. Despues en el backward pass, eso cancela el flujo de gradientes porque el grad seria cero tambien entonces rompe la cadena para esas rutas.

Apply noise:

En dropout ya agregamos ruido aleatorio a las activaciones de la red. Entonces donde mas podemos agregar ruido para generalizar:

- Input data: ayuda. Tambien se puede buscar el worst-case additive noise (adversarial training).
- Weights.
- Labels (label smoothing). Le cambiamos el valor del label a ciertos datos asi reducimos overconfidence.

Otras tecnicas para mejorar el rendimiento:

- Transfer learning: tomar un modelo pre-entrenado, que suponemos que va a tener buenas representaciones aprendidas, y hacer como que inicializamos el modelo asi. Y despues le quitamos la ultima layer y le agregamos una nueva o varias y hacemos fine tuning solo de esas o del modelo entero para una nueva tarea.
- Augmentation: podemos crear mas data, haciendole cambios pequeños a la data que tenemos y asignandole el mismo label. Esas modificaciones al input original hacen que tengamos mas datos y que ayude a generalizar y no que se fije en pequeñas cosas que no son las importantes.

## Residual networks

El problema principal es que, a medida que la red se vuelve mas deep, empiezan a haber fenomenos que hacen que no sean faciles de estabilizar y entrenar, por ejemplo que al cambiar apenas el valor de las primeras layers, termina cambiando un monton todo lo demas y los gradientes cambian tambien un monton… entonces al estar todo tan inestable y riesgoso, hace que la suerte te haga terminar en cualquier lado.

Hasta ahora siempre fue todo secuencial. Pero podemos agregar residual connections.

**Residual connections**

Son BRANCHES en el grafo computacional, donde el input a cada layer se vuelve a sumar al output de esa layer.

$$
\begin{align*}\mathbf{h}_1 &= \mathbf{x} + f_1[\mathbf{x}, \boldsymbol{\phi}_1] \\\mathbf{h}_2 &= \mathbf{h}_1 + f_2[\mathbf{h}_1, \boldsymbol{\phi}_2] \\\mathbf{h}_3 &= \mathbf{h}_2 + f_3[\mathbf{h}_2, \boldsymbol{\phi}_3] \\\mathbf{y}   &= \mathbf{h}_3 + f_4[\mathbf{h}_3, \boldsymbol{\phi}_4]\end{align*}
$$

Ahora las h participan como input a la siguiente layer pero tambien sumandose de una.

Orden de operaciones:

En una red tipica, se hace un linear → ReLU (u otra funcion de activacion). Pero si hicieramos esto con residual connections, siempre SUMARIA algo…

![image.png](AI%20notas/image%2049.png)

Entonces para que no pase esto, lo que se suele hacer cambiar el orden:

1. Primero se aplica la funcion de activacion
2. Despues le sigue la transformacion lineal

![image.png](AI%20notas/image%2050.png)

Se suele primero empezar con una Linear transformation porque si despues tenemos relu, queremos permitir que los valores no sean todos negativos.

Tambien en un bloque residual se pueden tener varias transformaciones con linear y relu… y que despues se le sume el residual connection. O sea, que la conexion residual sea despues de varias layers.

**Residual networks permiten que se entrenen modelos mucho mas profundos porque permite el flow de gradientes mucho mejor.**

- **Sin residuales**: En redes profundas, las señales (gradientes) se "pierden" al pasar por muchas capas, como un mensaje en un teléfono descompuesto que se distorsiona hasta volverse inútil. El modelo no aprende bien y el rendimiento empeora.
- **Con residuales**: Agregan "atajos" que suman la entrada directamente a la salida de una capa (output = transformación + input). Así, la info fluye directo, los gradientes no se desvanecen, y puedes apilar muchas más capas sin colapso, como construir una torre con soportes internos que la mantienen estable.

![image.png](AI%20notas/image%2051.png)

# Attention en secuencias

**Secuencias en redes neuronales comun**

Las redes neuronales como las vimos sirven para inputs fijos. Que pasa si trabajamos con secuencias y queremos predecir el siguiente valor en una secuencia? Las secuencias en principio pueden ser de longitud variable, como en texto, las oraciones pueden tener distinta longitud. Podriamos usar paddings que es dejar en blanco mucho y concatenar las embeddings y listo pero esto hace perder mucha eficiencia.

Problemas:

- No reconocer los tokens: Todo el input flatenizado se trata como un solo conjunto de numeros sin individualizar las distintas palabras. Mezclaria todo y procesaria todo. Pensa que las posiciones en un MLP son fijas… un weight conecta el input i a la neurona j… entonces siempre hace desde el mismo input i a la neurona j, independientemente de si hay una palabra o la otra… entonces como podria modelar que una familia de palabras tengan mas peso en otra familia de palabras o en cierta logica de la oracion si no se individualizan las palabras? No tiene nocion de palabras.
Los pesos no se adaptan dinamicamente segun el contexto sino que quedan fijos para una posicion.
- Escalabilidad: flattenizar el input se convierte en algo enorme. Te rompe la memoria. Y con paddings se introduce muchisimo ruido y encima te inhabilita weights o neuronas si en el entrenamiento ve muchos.

Se probo muchisimo las MLP para secuencias pero no funcionaban bien.

**RNN (80s, 90s pero explotaron en deep networks)**

Entonces pensaron en como nosotros procesamos secuencias. Cuando leemos vamos recorriendo la secuencia y teniendo una memoria de lo que venimos leyendo. Eso podria representarse en un ESTADO latente, que vaya guardando la informacion de la secuencia.

La idea es que: cada palabra la convertimos en una representacion numerica (un vector de numeros → embedding). Vamos procesando uno a la vez y vamos TRANSFORMANDO ese vector de palabra en un hidden state (la memoria). Pasamos a la siguiente palabra y tambien la combinamos con la memoria para actualizarla… y asi para toda la secuencia.

1. Tenemos la palabra input actual → la multiplicamos con una matriz de weights para inputs.
2. Tenemos el hidden state (memoria) hasta el momento → la multiplicamos con una matriz de weights para hidden state
3. La memoria actualizada = activación(W_xh * input_actual + W_hh * hidden_previo).

Las matrices de weights son dos para toda la red. Se comparten, se usa el mismo W_xh para todos los inputs y la misma W_hh para el hidden state en todos los pasos de la secuencia.

El hidden state es un VECTOR que lleva la memoria acumulada resumiendo todo lo que va viendo en la secuencia, palabra a palabra. Despues se puede hacer una transformacion diferente para terminar en la prediccion con un softmax para el vocabulario.

Problemas:

**Vanishing gradients**: En una RNN usamos las mismas matrices de weights tanto para los inputs como para los hidden states, en todos los pasos de la secuencia. 

Cuando se hace backprop, cada weight recibe una contribucion (gradientes) de cada paso de la secuencia. Aprende de cada paso. Pero esas contribuciones no fluyen de la misma manera:

- Los pasos finales (cerca del output) logran pasarle gradientes de una manera buena, porque estan cerca de la loss y es mas directo llegar a ellos (el grad viaja por pocas operaciones).
- Los pasos iniciales tienen que viajar por muchas operaciones en cadena hasta llegar al weight. Y si las secuencias son muy largas, al hacer muchas multiplicaciones, si no se mantienen estables los gradientes) van a tender a disminuirse. Entonces cuando multipliquemos la derivada local por el grad que viene, se va a disminuir la fuerza de la contribucion.

Para solucionar esto, en LSTM se crearon mecanismos para manejar mejor la memoria con gates:

- forget gate: olvida partes irrelevantes multiplicando por 0-1
- input gate: agrega info nueva selectivamente
- output gate: decide que emitir

Cada gate es una mini red que actuan como interruptores. Esto permite que info importante "fluya" sin diluirse, porque el forget gate puede ser 1 para partes clave, evitando vanishing gradients.

LSTM:

Ahora tenemos 2 memorias: una la que le mostramos al modelo (h hidden) y otra que es interna del modelo y que es la de largo plazo o la menos dinamica (c). La idea es que hidden se alimenta de c. Tambien c se va actualizando: olvida cosas irrelevantes, agrega nueva info y decide que emitir (filtrar para conformar h).

Para hacer todo esto tenemos weight matrices y bias de distintos tipos:

- Input: W_xi, W_hi, b_i
- Output (va a hidden): W_xo (weights del input), W_ho (weights del hidden anterior), b_o
- Forget: W_xf, W_hf, b_f
- Candidate: W_xc, W_hc, b_c

Para cada paso, tenes:

- x_t: el input de ese t
- h_(t-1): el hidden anterior
- c_(t-1): la memoria profunda, mas estable

Y quiero obtener/actualizar: 

- c_t (mi nueva memoria)
- h_t (el hidden state que uso en el modelo)

Los mecanismos funcionan combinando (suma) las multiplicaciones del hidden anterior y el input actual con sus respectivos weights que corresponda (forget, etc) mas el bias que corresponda.

**En aprendizaje haces que los distintos weights aprendan una logica que al multiplicarlo por hidden anterior e input actual generen un output que corresponda a su funcion de actualizacion a la memoria**. Son COMPUERTAS.

- Para forget: El resultado se pasa por sigmoid y da un vector entre 0 y 1. Dice que tan fuerte se conserva cada parte de la memoria c_(t-1).
- Para input + candidate: se hace en dos etapas:
    - Primero se propone la memoria **candidata g**. Se usa tanh y genera un vector candidato con nueva info para escribir en la memoria
    - Despues el **input gate** con sigmoid te dice cuanto va a entrar, con vector entre 0 y 1. Los multiplicamos para ponderarlos
    
    O sea primero se genera la informacion a añadir y despues se genera el ponderamiento o importancia de que añadir. 
    
- c_t entonces es = ( f_t * c_(t-1) )  +  ( i_t * g_t ). Mezclamos lo que decidimos mantener con lo que decidimos agregar
- output gate: usa sigmoid para decidir que “decir” o mostrar en el hidden para continuar al siguiente paso.

El h_t final es = o_t * tanh(c_t). Esto filtra que exponer.

**INTUICION DE DISEÑO** 

Esta buena esta forma de pensar de que las operaciones que vos haces con distintos significados las aprende la maquina con su matriz de weights. El “diseño” que uno le da para que opere pasa por el lugar que uno le da en la actualizacion final. Despues cada weight aprende como hacerlo solo.
**Solo le damos la estructura.** Le das un lugar/espacio para que haga cosas, un sistema para tomar decisiones estructuradas.

**Asignar roles distintos a distintos grupos de pesos.** Y asegurarte de que tengan **espacios de decisión independientes**, aunque estén entrenados juntos.

**Separás el flujo de información en caminos diferentes.** A cada camino le das su propia **matriz de pesos**, y su propia **salida**. 

Le decís a cada camino: “Vos vas a decidir X”. Pero no lo programás con reglas. 
**Solo lo ponés en posición de aprender a decidir X, mediante un lugar específico en la fórmula final.
Diseñás la ubicación donde esa lógica va a afectar.**

```jsx
c_t = f_t * c_{t-1} + i_t * g_t
```

Es pensar: este output afecta a que? Y como seria el resultado de todo el conjunto segun el comportamiento de ese output.

- f_t esta multiplicando a c_(t-1) y es entre 0 y 1, entonces **se le da el poder de olvidar o no**
- i_t * g_t → es algo que viene de afuera y es parte de la nueva memoria. Se le da el poder de meter informacion a la memoria. A su vez, i_t es entre 0 y 1 entonces le das el poder de cuanto de g_t (candidato) entra
- h_t = o_t * tanh(c_t) → se le da a o_t el poder de decidir que exponer

ESTO ES DAR ESTRUCTURA, **decidir cómo se usa el output de cada grupo de pesos**, y **en qué punto del cálculo aparece.**

**ES MODELAR LOGICA DE COSAS O SISTEMAS**, dejando que aprenda los parametros de cuanto.

**Importa la posicion que ocupan en la arquitectura**. 
El **significado funcional** de cada grupo de pesos lo define **la forma en la que sus salidas se usan**.

**Diseñar arquitecturas neuronales es modelar la *lógica estructural* del sistema que queremos que la red aprenda.**

Declaramos *qué tipos de información existen* (memoria lenta, estado rápido, eventos), *cómo interactúan* (olvidar, agregar, exponer), y *dónde entrar* en las ecuaciones.

**Los pesos aprenden sólo “cuánto” realizar cada operación**, porque su salida afecta directamente la actualización de esos estados y la pérdida retropropaga la presión correcta.

Las Gated Recurrent Units (GRU) lo simplificaron solo con update y reset gates.

![image.png](AI%20notas/image%2052.png)

INTUICION DE RNN GENERAL

https://karpathy.github.io/2015/05/21/rnn-effectiveness/

Son mucho mas poderosos que las redes convencionales porque no estan limitadas a un tamaño fijo de pasos de computo ni a vectores de i/o de tamaño fijos.

**RNN son Turing-Complete**: Pueden simular cualquier programa arbitrario (con los weights adecuados). EN TEORIA…
Esto es porque vos combinas el input vector con un state vector con una FUNCION fija pero aprendida, y eso produce un nuevo state vector. Entonces es como que aprende una funcion (programa) que tiene inputs y variables internas.

Muchas cosas se pueden **modelar como secuencias**.

**Deep RNN**: Los RNN pueden ser deep tambien. La salida de una capa es la entrada de otra:

![image.png](AI%20notas/image%2053.png)

Aca hay una sola hidden layer pero podrian haber mas:

![image.png](AI%20notas/image%2054.png)

O visto de otra forma:

![image.png](AI%20notas/image%2055.png)

El hidden vector antes iba solo al siguiente step… pero si es profunda ahora va a:

1. Siguiente layer → el hidden state vector de una capa entra como el input de la siguiente hidden layer
2. Siguiente step, como antes

**Secuencialidad**: Por la naturaleza recurrente, no se puede paralelizar las operaciones sino que se tienen que hacer de a uno. Eso es un problema porque las GPUs no van a darte beneficios en este caso.

**Hidden state bottleneck**

El mayor problema sigue siendo que toda la **memoria es simplemente un vector**, entonces no hay tanto que se pueda representar en un vector. Si hacemos muy largas las secuencias o muy complejas, como vamos a representar todo eso con un vector? Perdemos mucha informacion. Tenemos como que usamos una lista de numeros para resumir toda una secuencia que puede ser super compleja. Hay que comprimir demasiado y perder detalle. Toda la info historica se aplasta en ese espacio limitado. Si la frase tiene 5 palabras o 50, todo debe ser comprimido en un único embedding. Esto limita la capacidad de la red para representar con precisión todo el contexto. **No se puede acceder directamente a distintas partes del input original** sino que contamos solamente con ese resumen chiquito comprimido.

**Seq2Seq (encoder-decoder)**

Para tareas como traduccion, se quieren mapear una secuencia a otra (seq2seq). Pero no es uno a uno sino que la longitud de las secuencias varian. 

Para eso se desarrollaron las arquitecturas **encoder-decoder**.

- **Encoder**: una RNN que procesa la secuencia input paso a paso y genera un **hidden state** final, que representa toda la secuencia.
- **Decoder**: otra RNN que usa ese vector del encoder como starting point (hidden inicial) y va generando tokens autoregresivamente, prediciendo las palabras de a uno de la secuencia de output (la traduccion por ej). Tambien va generando su propio hidden state pero tomando como punto inicial el hidden state del encoder.

PERO SIGUE TENIENDO EL MISMO PROBLEMA → **BOTTLENECK EN EL HIDDEN STATE**. 

Se sigue necesitando una forma de, **para cada paso en la generacion, poder mirar todo el contexto anterior**, como lo podemos hacer nosotros. 
En la traducción humana, uno **no memoriza por completo** una oración larga antes de empezar a traducir; más bien, se **va mirando la oración original** conforme se produce la traducción, enfocando la atención en la parte relevante del texto fuente para cada fragmento que se traduce

**Attention (primeras ideas)**

En vez de meter toda la secuencia en un solo vector, **por que no permitir que el decoder mire directamente a cada token del encoder?** Bahdanau attention en 2014 para traduccion.

**En vez de un context vector fijo (hidden final), el decoder, en cada paso de generación, "atiende" dinámicamente a todos los hidden states del encoder (no solo el último)**.

El tema es: el decoder, en cada paso, puede acceder al hidden state del encoder y ver que hay ahi… 
Pero COMO ELEGIR que hidden states mirar? Algunos le sirven mas que otros no? Pero como sabemos? 

La idea es permitir que la red **aprenda a *alinear* la salida con partes específicas de la entrada** durante la traducción. Este mecanismo de alineamiento aprendido es lo que llamamos **mecanismo de atención**. 

La idea central es que el decoder, en cada paso, **identifique (o “busque”) qué parte de la secuencia de entrada es más relevante** para producir la siguiente palabra de salida, en lugar de confiar en una única representación fija de toda la entrada.

Esa **ALINEACION es encontrar un PUNTAJE de que hidden state es mas relevante para ese momento puntual de generacion,** 

Las **puntuaciones se obtienen comparando** de alguna forma el estado actual del decoder con cada estado del encoder (modelo de alineamiento / atencion).

Y con eso, **generar un vector contextual con lo que se necesita en ese momento**, no de todo en general. Se forma combinando todos los hidden states pero dandole mas atencion a unas partes que a otras.

Se puede hacer de distintas maneras:

- En primer lugar (Bahdanau). En cada paso del decoder, se calcula una puntuacion para cada hidden state del encoder mediante una red feedforward simple de una capa para calcular una **funcion de puntaje** que estima que tanto deben alinearse. Despues se usa softmax sobre las puntuaciones para obtener **pesos de atencion**.
El decoder calcula el **vector de contexto como el promedio de los hidden states del encoder, ponderado por los pesos de atencion**. 
Este vector es esencialmente una especie de “vista enfocada” de la oración original, una combinación lineal de las representaciones del encoder, enfatizando más las posiciones consideradas relevantes.
Ese vector contextual se usa, junto con el estado recurrente del decoder, para predecir la siguiente palabra. Para el proximo paso de decoder, se vuelven a calcular las atenciones y obtener un contexto nuevo.
El decoder ahora tiene acceso directo a todos los estados intermedios del encoder.
- Despues simplificaron, haciendo la **similitud como un dot product entre query y key**.
    - Query: es el hidden state del decoder RNN (el momento en el que estamos prediciendo) que ya es un vector con memoria acumulada hasta el momento. 
    Puede representar: **lo que necesito encontrar en el input**.
    - Keys: son los hidden states del encoder (son vectores que representan informacion de cada palabra mezclada con el contexto hasta ese momento).
    Puede representar: **identificador del contenido a cada posicion de la oracion input. Lo que ofrece**.
    
    Dot product porque mide la similitud para cada componente (dimension) entre dos vectores. Si estan alineados, el valor es mayor. Es mas rapido porque no hay una red aparte sino una multiplicacion.
    En las primeras versiones:
    
    - Query → es el decoder hidden state actual
    - Keys → hidden states del encoder (una para cada palabra con su contexto)
    - Value → hidden states del encoder (lo mismo que keys)
    
    Keys se usaban para dos cosas:
    
    1. Como Keys → Calcular la similitud (con el dot product) 
    2. Como Value → es el contenido en si, los hidden states del decoder que se van a combinar linealmente, ponderado por la similitud o atencion.

![image.png](AI%20notas/image%2056.png)

# Transformer architecture

## Self-attention

**Transformer (Attention is all you need - Vaswani 2017)**

**Self-attention sin RNN, vectores Q, K, V**

Chau RNNs:

Desde bahdanau ya medio que quedo solucionado el tema de la inexpresividad de la memoria como hidden states, porque ahora en cada momento se puede mirar todo el input anterior. Ahora es mucho mas flexible la memoria

Otro problema de las RNNs es que no son paralelizables entonces estaria bueno poder aprovechar GPUs para hacer todo mas rapido. Si las pudieramos eliminar estaria bueno.

Para que teniamos RNNs supuestamente? Para ir generando un contexto que involucre todo lo que vamos viendo… pero si ahora tenemos ATTENTION, que puede ver todo lo anterior, para que queremos construir un hidden state lleno de contexto, secuencialmente? Podriamos ir a mirar TODO el input y agarrar lo que nos interese.

NOTA: No se si es exactamente lo mismo porque lo bueno de lo secuencial es que en cada hidden state ya estas metiendo info de todo… entonces es la representacion de esa palabra en ese contexto. No es lo mismo que mirar la palabra sola aislada.
Las palabras solo aisladas no capturar dependencias ni orden…

La idea podria ser que el contexto se CONSTRUYA directamente en attention, en vez de en los hidden states del encoder. 
Y el tema del orden que se resuelva de otra manera. Esto se puede resolver con OTRO EMBEDDING QUE INDIQUE LA POSICION (sumarle al vector otro vector de posicion, que le agregue esa informacion).

**SELF-ATTENTION**

Entonces como capturamos el contexto? 

1. Cada token tiene su ID, que es el embedding combinado con su positional embedding.
2. Estamos en el ultimo token de la secuencia. Hay un vector Q que transforma nuestro vector ID de embeddings+position en una query (tipo de informacion en el que estamos interesados) → individual
3. A la vez, a todos los tokens (a sus IDs embeddings+position) los transformamos usando vector K en keys, que es un identificador del tipo de informacion que tiene cada token → individual
4. Con la query del ultimo token, miramos a todos los keys de tokens de la secuencia y hacemos un dot product → los que mas matchean en sus componentes va a tener un valor mas alto. Esto es la AFINIDAD QUE TIENEN. Despues se hace softmax para crear los scores de attention.
5. Para cada token, se multiplica por otro vector V que transforma el ID+position en value, que es la informacion que se quiere transmitir. Se usan los scores de attention para ponderar y se hace un promedio ponderado de los value → **ESTO ES EL CONTEXT VECTOR** CON SELF-ATTENTION.

Ese context vector es una forma de replicar el hidden state que tendriamos en RNN pero sin hacerlo secuencial. Tiene ventajas:

- Es full paralelizable
- No tiene problemas con gradientes porque el grafo de operaciones ahora no es tan largo sino que es mas directo

POR QUE Q, K, V distintos?

En realidad la idea inicial es: yo quiero incorporar la informacion de algunos tokens del pasado a mi nuevo contexto actual. 
Como hago para prestarle mas atencion a la info de unos tokens mas que a otros? Como lograr saber a cual mas atencion y cual menos?

La idea es terminar con un promedio ponderado → sum( value * attention_score). El values es el contenido del token y el attention_score es cuanto nos interesa ese token. 

Para saber cuanto nos interesa un token, desde otro token particular, es logico pensar en una forma de ver si hay mas o menos match. Si lo que necesitamos es parecido a lo que nos ofrecen. Queremos:

1. Algo que busque (query): Informacion sobre el momento en el que estoy y lo que necesito
2. Algo que represente lo que hay (key): para cada token, se muestra que ofrece

EJEMPLO: https://github.com/greentfrapp/attention-primer/blob/master/1_counting-letters/README.md#model

Si quiero calcular el precio promedio de bebidas:

- Query: “drinks” (podria ser que busca ciertos items, sustantivos, etc)
- Keys: son los nombres de los productos (es una palabra que puede significar muchas cosas pero entre esas un sustantivo, puede verse como un item, etc)
- Value: el precio del producto. Es lo que nos interesa aportar. Es la info real para despues hacer calculos

https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms?rq=1

The difference from the above figure is that the queries, keys, and values are **transformations** of the corresponding input state vectors. The others remain the same.

What are the benefits of this matrix multiplication (vector transformation)?

The obvious reason is that if we do not transform the input vectors, the dot product for computing the weight for each input's value will always yield a maximum weight score for the individual input token itself. In other words, when we compute the n attention weights (j for j=1, 2, ..., n) for input token at position i, the weight at i (j==i) is always the largest than the other weights at j=1, 2, ..., n (j<>i). This may not be the desired case. For example, for the pronoun token, we need it to attend to its referent, not the pronoun token itself.

Another less obvious but important reason is that the **transformation may yield better representations for Query, Key, and Value**. Recall the effect of Singular Value Decomposition (SVD).

See [Attention is all you need - masterclass](https://youtu.be/rBCqOTEfxvg?t=946), from 15:46 onwards Lukasz Kaiser explains what *q, K* and *V* are.

So basically:

- *q* = the vector representing a word
- *K* and *V* = your memory, thus all the words that have been generated before. Note that *K* and *V* can be the same (but don't have to).

https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms/551126#551126

https://www.reddit.com/r/MachineLearning/comments/bkw2xp/d_what_is_the_rationale_behind_selfattention/

If you're looking for a more intuitive explanation, I like to think of self-attention as a lookup table for vector spaces.

Similar to how one searches records in a database, a list of keys are scored with respect to some query. A large dot-product between the query and key vector means that the angular distance is small, and so results in a high activation. The mechanism would like to select those vectors that match the most, i.e., have the highest activation. Sometimes the mechanism may select one vector, and other times it selects them all. After selection, each value vector corresponding to the matched keys is weighted proportionally to the activations (after softmax normalization) and summed together.

CONCLUSIONES

**Attention funciona mejor que solo el hidden state de una RNN**

Porque es como una "memoria dinámica" que permite al modelo "mirar" selectivamente al pasado (o a toda la secuencia) en lugar de confiar en un estado fijo y comprimido como en las RNNs puras. Eso lo hace más flexible, porque no perdés info en un bottleneck, y tenés acceso directo a detalles originales, ponderados por relevancia.

**Attention con QKV funciona mejor que attention con hidden states**

(Aparte del paralelismo)

En pre-transformers, el Value ya tiene informacion enriquecida por el contexto y la Query tambien hace la pregutna contextual… A mi me suena que es super valioso.

Pero tambien QKV funciona bien porque logra separar utilidad y especializa mejor. K y V no tienen por que ser el mismo… La Query no tiene por que tambien usarse para predicciones futuras. Al separarlas das mas expresividad para que logren especializarse mejor para su uso.

- Q se optimiza para "preguntar" basado en el token actual (¿qué necesito buscar?).
- K para ser "comparable" (vectores que midan bien similitud, como dimensiones enfocadas en patrones clave).
- V para guardar "detalles puros" (info sin alterar para similitud).

Si usás el mismo vector (como en RNNs), perdés esta flexibilidad –es como usar un martillo para todo, en vez de herramientas especializadas.

- Tambien, no corren el riesgo de vanishing gradients y los problemas de los RNN.

INFO CONTEXTUAL DE ATTENTION → MULTIPLES CAPAS (SECUENCIALIDAD)

En realidad attention mechanism tiene secuencialidad → no tanta como RNN. 

**Hay multiples iteraciones de attention**. 
Tiene una especie de jerarquizacion al pasar por multiples capas… como en todas las redes neuronales.

- **Al inicio (capa 0)**: Inputs son embeddings independientes por token (+ positional encodings). 
Q, K, V se proyectan de ahí. 
Attention combina: Para un token i (su Q_i busca en K de todos, pondera V de todos).
Output_i es un mix inicial –ya tiene algo de contexto básico (e.g., "cat" atiende a "The" si scores altos).
- **Pero el contexto real surge en capas múltiples**: Cada capa toma el output de la anterior como input. 
En capa 1: Attention sobre representaciones ya "mezcladas" de capa 0. 
En capa 2: Sobre las de capa 1 (aún más enriquecidas). 
Es como una "recurrencia implícita" sin vanishing –cada capa "refina" el contexto global.

**Intuitivo**: Imaginate una oración: Capa 1 captura vecinos cercanos (sujeto-verbo). Capa 2 conecta frases distantes (e.g., pronombre al referente). Al final (6-12 capas típicas), cada token's representación lleva contexto profundo de toda la secuencia, pero sin compresión forzada.

La QUERY en realidad va a estar constituida solo de un token (el actual). Pero bueno, con su embedding + posicion, puede funcionar bien igual. 

## Transformer block

![image.png](AI%20notas/image%2057.png)

Attention block

![image.png](AI%20notas/image%2058.png)

---

EXPLICACION:

Input inicial:

Entra un token index al transformer. Eso se multiplica (one hot) con una matriz de embeddings de los tokens (son aprendibles). Para una secuencia queda un vector embedding por token. A eso se le suma el positional embedding. 

Attention block 1 (capa 1)

Para cada token, tomas el vector input x y lo multiplicas por tres matrices:

- Q¹ = X₀ × W_Q¹
- K¹ = X₀ × W_K¹
- V¹ = X₀ × W_V¹

Luego aplicamos attention:
Attention_output¹ = softmax(Q¹ × K¹ᵀ / √dₖ) × V¹

Esto nos da un nuevo vector para cada token.

MULTI-HEAD ATTENTION 

No solo se hace una vez esto sino N veces. O sea, tenemos N diferentes W_Q, W_K, W_V. P (son mas chicos, es como que se particionan, no son todos full).

Luego de que se hace N veces, se concatenan los resultados para cada token y se hace una proyeccion lineal. Entonces vuelve a tener el mismo shape que antes el vector de cada token.

Add & Norm

Se hace un residual connection con el input original, que se suma a la salida del multi-head attention. 
Y se normaliza.

FeedForward

El output de eso se pasa por una pequeña red feedforward para “computar” CADA TOKEN EN PARTICULAR, individual. Esto sirve para transformar no contextualmente sino modelar ya la info que tiene cada token individual. 

Para cada attention block va a haber un feedforward con weights diferentes.

Attention block 2 (capa 2)

Ahora en lugar de los embeddings originales + posiciones, usamos los vectores que salieron de la capa anterior!

Y se procesa lo mismo…

## Transformer architecture

TRANSFORMER ORIGINAL ARCHITECTURE (Encoder-decoder)

![image.png](AI%20notas/image%2059.png)

**GPT architecture (Decoder-only)**

![image.png](AI%20notas/image%2060.png)

# Eficiencia

## GPU basics

https://jax-ml.github.io/scaling-book/gpus/

Es una computadora que tiene una arquitectura con:

- Muchos compute CORES (streaming multiprocessors → SMs) simples, especializados en multiplicacion de matrices
- Conectados a una memoria rapida (HBM)

![image.png](AI%20notas/image%2061.png)

Cada SM tiene:

- Tensor cores: dedicados a matmuls de baja precision (FP16, 32)
- Vector arithmetic unit (warp scheduler + CUDA cores): CUDA core → procesador escalar simple (pointwise operations sobre escalares o vectores)
- L1 cache (SMEM)

Despues tambien hay otras memorias:

- L2 cache
- DRAM (High bandwidth memory HBM): se guardan parametros, activaciones, optimizer states, etc

Una GPU moderna tiene mas de 100 SMs. Cada SM es mas o menos independiente, por eso se pueden hacer cientos de tareas al mismo tiempo.

SM:

![image.png](AI%20notas/image%2062.png)

Tienen 4 cuadrantes → subparticiones. Cada uno tiene:

- 1 tensor core (matmuls)
- 16k 32-bit registers
- SIMD/SIMT vector arithmetic unit (warp scheduler), cuyas lineas se llaman CUDA cores

**CUDA cores:**
Cada subparticion tiene un conjunto de ALUs lalmados CUDA cores, que hacen SIMD/SIMT vector arithmetic. 
Son: 32 fp32 cores y algunos menos int32 cores y fp64 cores.
Todos ejecutan la misma instruccion en cada ciclo. Si un core esta sumando dos floats, todos los demas tambien lo hacen.
Hacen los ReLUs, pointwise vector operations y reductions (sumas).
Usan SIMT (single instruction multiple threads), en comparacion de otros modelos como SIMD (single instruction multiple data).

**Tensor core:**
Cada subparticion tiene 1 Tensor Core, dedicado a matmuls. Representa la **mayoria de los FLOPs**.
Tambien pueden hacer matmuls de menor precision a mayor throughput (el doble para fp8 que para fp16).

**Memory:**

Tiene una jerarquia de memorias: HBM (main GPU memory), Caches (L2, L1/SMEM, TMEM, register)

- Registers: tienen 16384 palabras de 32 bits (256kb por SM). Son accesibles por los CUDA cores.
- SMEM (L1 cache): Cada SM tiene su propio 256kB on-chip cache. Aca se gaurdan activaciones e inputs a los tensor core matmuls
- L2 cache: todos los SMs comparten este cache de 50MB (bastante grande). Se usa para reducir los accesos a memoria principal.
- HBM: es la memoria principal del GPU. Se usa para guardar model weights, gradients, activations, etc. Puede ser de varios GB (20, 32, hasta 192 en las ultimas).
    - El bandwidth desde GBM a los CUDA tensor core se llama **memory bandwidth** o HBM bandwidth (3.35TB/s on H100).

## Model arithmetic, problemas & bottlenecks

OBJETIVO:

Hacer que tu modelo (dentro de los recursos que tenés):

- sea lo más grande posible (capacidad),
- corra lo más rápido posible (velocidad),

Pregunta clave: **¿Qué me permite y qué me limita mi hardware?**

- **Entra el modelo**?
- **Donde se va el tiempo**?

RECURSOS:

- **Memoria**: (VRAM): cuantos datos puedo guardar **al mismo tiempo** (bytes/s). 
Si me paso → OOM.
- **Bandwidth**: Que tan rapido puedo **mover datos hacia y dentro de GPU** (GB/s).
Puede ser el bottleneck si necesito muchos datos para hacer unos pocos calculos.
- **FLOPs**: Cuantas operaciones matematicas puede hacer la GPU (TFLOPs/s).
Si hay muchas cuentas para hacer, es el bottleneck.

> L (capas), d (dimension), f (dimension FF), T (seq length), B (batch)
> 

---

**MEMORIA** → Entra el modelo?

Para el entrenamiento, durante el forward, backward y optimizacion, hay distintas cosas que pasan en memoria:

- Persistentes (estan en toda la corrida, en forward y backward)
    - Parametros
    - Estados del optimizador (m y v en Adam)
    - Buffers, estadisticas de capas (norms), etc
- Step (se crean, usan y liberan cada step)
    - Batch (data o la porción que la primera capa guardó para backward)
    - Activaciones del forward (guardo lo que necesito recordar para el backward… lo demas lo computo, lo uso para la siguiente capa y lo elimino)
    - Gradientes en el backward

**Paso a paso** (step: forward → backward → update)

- Antes de empezar, ya estan en la GPU:
    - Parametros
    - Estados del optimizador
1. Se carga el batch de data
2. Forward: se calculan las salidas de cada capa. Se guarda lo minimo necesario para usar en el backward despues.
    1. Se van guardando para cada layer cosas de las activaciones. Se van acumulando. Las que no se necesitan no se guardan (solo se calculan como temporal y se liberan)
    2. Peak: fin del forward donde tenes muchas cosas guardadas + parametros + estados del optimizador + batch (la parte que el backward necesita)
3. Backward: va calculando los gradientes de los parametros de cada capa y mientras se van liberando las activaciones que habian quedado del forward cuando se usan.
    1. No se guardan todas las matrices de gradientes intermedias (de las activaciones o algunas otras cosas que se calculan) sino que solo va fluyendo hacia atras y SI SE GUARDAN LOS GRADIENTES DE LOS PARAMETROS W,b del modelo.
    2. En el medio del backward tenes: gradientes intermedios (algunos ya liberados, otros calculados), muchas de las activaciones guardadas en el forward pero que todavia no se usaron en el back, gradientes guardados de los parametros del modelo hasta esa layer, etc (ademas de parametros, estado del optimizer, batch)
    3. Peak: al comienzo del backward porque todavia estan casi todas las activaciones pero ya hay algunos grads de params. Despues la memoria baja cuando se liberan activaciones. Al final quedan las matrices de grads de los parametros del modelo que se suman.
4. Fin del backward: Ya no hay activaciones. Estan todos los grad de los params.

**Como escala**

Parametros

- Attention: 4 d**2 por capa
- MLP: 2 f d**2 por capa
- Total por capa: (4+2f) d**2
- Total modelo: L (4+2f) d**2
- Embeddings: vocab * d
- Bytes → hay que multiplicarlo segun dtype (bf16 = 2 bytes)

> Crecimiento:
> 
- d escala CUADRATICAMENTE
- L escala linealmente

---

**TIEMPOS** → donde se va el tiempo?

Objetivo → Aumentar la cantidad de FLOPs/s que hacemos posta. Que la GPU este ocupada todo el tiempo haciendo FLOPs (saturarla). Lo ideal es que sea:

- **Compute-bound**
- **FLOPs cerca del pico ideal** del GPU

Cualquier otra cosa que interrumpa esto, es ineficiente o un bottleneck.

**Componentes** de tiempos:

- **Input pipeline**: lee datos desde el disco a CPU, descomprime, transformaciones, arma el batch y lo transfiere a GPU)
Bottlenecks:
    - Disco lento
    - Pocos workers (DataLoader), no usar pin_memory, no usar non_blocking
    - Dataset muy pesado
- **Comms dentro de GPU**: desde VRAM hacia L2, registros, etc. Lectura de weights, activaciones, outputs intermedios, etc
Bottlenecks:
    - Modelos con baja arithmetic intensity (muchos datos, pocas cuentas)
    - Cuando se hacen muchas escrituras/lecturas a memoria o hay muchas esperas por la arquitectura del modelo
    - No reutilizar bien los datos en memoria (cache)
    
    Estos son **memory-bound**.
    
- **Computo (FLOPs)**: La GPU hace multiplicaciones y sumas. Queremos maximizar esto y que llegue hasta el maximo ideal.

**Memory-bound vs Compute-bound** → dentro de GPU

```python
Arithmetic intensity = FLOPs / bytes movidos
```

Se compara con un valor critico que depende del hardware. Si estas por debajo, memory-bound. Si no, compute-bound.

**Problemas**:

- Diseño del modelo (arquitectura)
- Configuracion del entrenamiento
- Uso de memoria inteligente

Si hacemos muchas operaciones con pocas info → compute bound

Si hacemos pocas operaciones y hay que leer y escribir mucha info → memory bound

**Como escala**:

Attention → costo cuadratico. 

- Linear projections (Q,K,V) → FLOPs: 3 B T D D = 3 B T D**2
- Atencion (Q @ K.T) → FLOPs: B T**2 D
- Softmax → FLOPs: B T**2 D
- Proyeccion final → FLOPs: B T D D
- TOTAL FLOPs: B T D**2 + 2 B T**2 D
- Escala cuadraticamente tanto en D como en T (T es mas importante porque puede ser arbitrario)

MLP

- FLOPs: 8 B T D**2
- Cuadratico en D. Lineal en T → pero puede ser mas grande que attention si T no es muy largo.

En **FLOPs, attention vs MLP**: 

- Attention: T**2 * D
- MLP: T * D**2

Es dominante attention si T >> D

- Si `T = 4096` y `D = 512` → gana attention.
- Si `T = 512` y `D = 2048` → gana MLP.

---

**TRANSFORMERS → ESCALA** (memoria, flops, bandwidth (interno y externo)

| Componente | Memoria | FLOPs | Comm. Interno (bandwidth) | Dominancia / Costo |
| --- | --- | --- | --- | --- |
| **Embeddings** | `T × D` (pequeño) | despreciable | bajo | 🟢 despreciable |
| **Attention (QKV)** | `T × D` + pesos (`D×D`) | `B × T × D²` | medio | compute-bound.
Crece en D |
| **Attention (QKᵀ)** | `T × T` intermedio | `B × T² × D` | alto | memory-bound.
FLOPs crece en T |
| **Attention (softmax @ V)** | `T × D` | `B × T² × D` | alto | memory-bound |
| **Attention (proyección final)** | `T × D` | `B × T × D²` | medio | compute-bound.
Crece en D |
| **MLP (D → 4D → D)** | `T × 4D` intermedio | `B × T × D² × 8` | medio | compute-bound.
Crece en D |
| **LayerNorm / activaciones** | `T × D` | despreciable | bajo | FLOPs despreciables |
| **Batch (input)** | `B × T` | 0 | bajo | ⚠️ depende del loader |
| **Gradientes** | igual que pesos | 0 (solo memoria) | moderado | ⚠️ relevante |
| **Optim states (Adam)** | igual que pesos | 0 (solo memoria) | alto (R/W) | memory-bound |

Dominan en memoria:

- Activaciones (T*D) por cada layer: L * B * T * D
- Parametros
- Estados del optimizador

Dominan en FLOPs:

- MLP si es grande D y es chico T
- Attention si T es grande (por el T**2)

Dominan en comunicacion:

- Input batch (mover a GPU)
- QKV → acceder a pesos dentro de GPU

**Memory-bound: Q@K.T**

La atencion suele ser memory-bound porque se trabaja por head (que tiene dimension chica).

- Un head tiene D_head = D/H
- FLOPs por head: 2 * T** 2 * D_head
- Memoria: tiene que leer Q y K, materializar S = QK.T de tamaño TxT. Escribir T**2 y leer T**2 para softmax
- Intensidad por head →  (2*T**2*D_h) / (c*T**2) =  (2*D_h)/c → como D_h es chico y c (numero de pasadas/lecturas) no lo es, la intensidad queda baja.
- Por head, el computo (con D_h chico) no alcanza para pagar los muchos bytes que se mueven → memory-bound.

**Compute-bound: MLP y Q,K,V,O projections**

Son matmuls grandes

- Bytes: se lee un bloque de weights y se reusa sobre muchas filas de activaciones
- FLOPs, por cada, forward:
    - Q,K,V,O proj: 4 * B * T * D**2
    - MLP: 2 * B * T * D**2
    - Intensidad: con D grande llenas los tensor cores y se hace compute-bound

---

**Estimar Memory, FLOPs y bandwidth** - comparar con GPU

**Memoria**

Buscamos solo el peak

**FLOPs**

Despues:

- Como puedo calcular:
    - Memoria: cuanto va a ser la max memoria usada (fin forward, inicio backward) a mano
    - Tiempos:
        - FLOPs: total, flops/s peak, etc… QUE DEBERIA MEDIR?
        - Bandwidth: total bytes movidos, peak, etc… QUE DEBERIA MEDIR?
        Como lo veo? Que conviene ver linea temporal? Peak? Cantidad de tiempo que estuvimos compute bound vs memory bound?
        - Que esta bien o mal?

[https://chatgpt.com/share/68a60c79-fcf0-8001-a3be-19b4f23f90b2](https://chatgpt.com/share/68a60c79-fcf0-8001-a3be-19b4f23f90b2)
raschka ch04/02
terminar [https://jax-ml.github.io/scaling-book/transformers/](https://jax-ml.github.io/scaling-book/transformers/)
https://jax-ml.github.io/scaling-book/transformers/
https://blog.eleuther.ai/transformer-math/
https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8
https://rohitbandaru.github.io/blog/Scaling-Deep-Learning/
https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4

## Trucos para eficiencia

- GPU, FLOPs, etc 
https://jax-ml.github.io/scaling-book/
https://jax-ml.github.io/scaling-book/gpus/
- Mejoras simples en eficiencia https://towardsdatascience.com/pytorch-model-performance-analysis-and-optimization-10c3c5822869/
- MFU https://medium.com/better-ml/using-model-flops-utilization-mfu-7b17de07faec

En general:

https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
https://towardsdatascience.com/pytorch-model-performance-analysis-and-optimization-10c3c5822869/
Ultra scale training playbook https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview

**Data loading**

Dataset + DataLoader: 
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

- num_workers
- prefetch_factor
- pin_memory
- non_blocking

**torch.compile**
https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
https://docs.pytorch.org/tutorials/recipes/compiling_optimizer.html

Lo compila primero, fusionando operaciones y generando kernels optimizados.

Hace menos llamadas entre python y CUDA.

Hay una latencia de compilacion.

**Flash attention**
https://pytorch.org/blog/flashattention-3/
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html

[https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)
torch.nn.functional.scaled_dot_product_attention

**Mixed precision (AMP)**
https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

**Distributed Data Parallelism (DDP)**
https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=distr_landing&utm_medium=intermediate_ddp_tutorial

**Pre-training advanced features** (LR warmup, cosine decay, gradient clipping)
https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-D/01_main-chapter-code/appendix-D.ipynb

# LLM Pre-training

## Datasets

- 

## Tokenizers

- karpathy https://github.com/karpathy/minbpe
- https://github.com/di37/gpt-tokenizer
