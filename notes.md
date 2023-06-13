# Logistic Regression

## ¿Qué es la regresión logística?

Es un algorito supervisado que pertece a los algoritmos de clasificación. Este no busca entregarnos un valor conituo si no que buscar entregar valores de 0 o 1. Es importante la desambiguación porque es cierto que en regresión lineal buscamos predecir valores continuos. La regresión logistica en su corazón se vale de la función sigmoidal, que va desde 0 a 1, un intervalo probabilistico para nuestras observaciones.

¿Qué probabilidad tiene una observación o dato de estar en 0 o en 1?
Podemos hacer una gráfica de Probabilidad de Aprobar vs Horas de estudio, y graficor algunos puntos de datos de estudiantes que practican desde 0 a n horas, cada punto de datos se puede mapear a la función sigmoide para mapear una probabilidad tal que:

Si está entre [0.5-1] el estudiante tiene mejores chances de éxito.
Si está entre [0-0.5) el estudiante tiene menos chances de ganar.
Manjenado los intervalos de probabilidad podemos realizar clasificaciones binarias de si y no.

## Tu primera clasificación con regresión logística

https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html

https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html

## ¿Cuándo usar regresión logística?

### La Regresión logistica se utiliza en los siguientes casos:

- Clasificación binaria: la regresión logística se utiliza para clasificar la observación en dos categorías distintas, como “sí” o “no”, “éxito” o “fracaso”, “compra” o “no compra”, etc.

- Datos de entrada no lineales: la regresión logística puede manejar datos de entrada no lineales y se puede utilizar para modelar la relación entre variables predictoras y variables de respuesta no lineales.

- Datos con valores atípicos: la regresión logística es robusta a los valores atípicos y no se ve afectada significativamente por los puntos de datos que se desvían de la tendencia general.

- Problemas de clasificación multiclase: si bien la regresión logística es una técnica de clasificación binaria, también se puede utilizar para problemas de clasificación multiclase mediante la técnica “uno contra todos”.

### Sintaxis
La sintaxis de sklearn permite configurar la distribucion de probabilidad o el metodo para resolver el problema
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

### Ventajas

- Fácil de implementar: Con sklearn es muy sencillo
- Coeficientes interpretables: Puedo entender los coeficientes y ver cómo se aplican.
- Inferencia: Puedes utilizar distintos features para ver cuál tiene más importancia para predecir tu variable dependiente.
- Clasificación en porcentajes: **No dice Sí y no, te da el porcentaje exacto de confianza {0,100}

### Desventajas

- Asume la linealidad: Asume que todas las relaciones son lineales y no siempre es así.
- Overfitting: Si pongo muchas features, este se aprende el patrón de entrenamiento en lugar de predecirlo.
- Multicolinealidad: Dos características que tienen el mismo comportamiento
- Datasets grandes: necesita datasets muy grandes para ser precisos, de lo contrario no alcanza a extraet toda la información.

### ¿Cuándo usarla?
- Sencillo y rápido: Cuando esto se busca.
- Prob de ocurrencia de un evento.
- Datasets linealmente separables.
- Datasets grandes.
- Datasets balanceados.

<img src="img/linear_vs_logistic.png" alt="data_engineer_gpc_logo" width="500"/>

## Fórmula de regresión logística

Los “odds” (en español, “cuotas” o “probabilidades”) son una forma de expresar la probabilidad de que ocurra un evento. En particular, los “odds” representan la relación entre la probabilidad de que ocurra un evento y la probabilidad de que no ocurra.

Por ejemplo, si la probabilidad de que un equipo de fútbol gane un partido es del 60%, entonces la probabilidad de que pierda es del 40%. En términos de “odds”, la probabilidad de ganar se puede expresar como 3 a 2, lo que significa que por cada 2 veces que pierde el equipo, gana 3 veces. De manera similar, la probabilidad de perder se puede expresar como 2 a 3, lo que significa que por cada 3 veces que gana el equipo, pierde 2 veces.

Los “odds” se utilizan comúnmente en las apuestas y en los juegos de azar, donde se usan para determinar las ganancias potenciales de una apuesta. En la estadística, los “odds” se utilizan en la regresión logística para modelar la relación entre las variables independientes y la variable dependiente binaria.

<img src="img/formula_log_reg.png" alt="data_engineer_gpc_logo" width="500"/>
