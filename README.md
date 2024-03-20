# Henry_PI_SteamGames
Desarrollo de un MVP para una API para un sistema de recomendacion y que analizará datos de reseñas de juegos de Steam.

# PROYECTO INDIVIDUAL Nº1
# Machine Learning Operations (MLOps)


# Descripción del problema 
Steam, es una plataforma multinacional de videojuegos. Con los datos de esta plataforma se desea crear un sistema de recomendacion de videojuegos para usuarios. Sin embargo, los datos no se encuentran en un formato adecuado para manipularlos, ya que son del tipo row y estan anidados en algunos casos, lo que dificulta el trabajo. El objetivo es obtener un MPV para este sistema de recomendacion.


Nota que aquí se reflejan procesos, no herramientas tecnológicas. Haz el ejercicio de entender qué herramienta del stack corresponde a cada parte del proceso

# Descripcion de tareas realizadas
## Transformaciones:
Los datos se proporcionaron en un formato similar a JSON, aunque algunos archivos resultaron más complicados de procesar que otros.
En primer lugar se subieron a Drive los documentos comprimidos. Para todos los conjuntos de datos, se utilizó la librería gzip para descomprimir los archivos comprimidos y acceder a su contenido. Para los datos de juegos de Steam, se utilizó la función json.loads() para convertir el texto en formato JSON en objetos de Python. En cambio, para los datos de reseñas y elementos de usuario, fue necesario utilizar una función de la librería ast para convertir cada línea en un diccionario de Python. Una vez convertidos, los datos se transformaron en dataframes para facilitar su manipulación y análisis en Python.

## Feature Engineering:
En el dataset user_reviews se incluyen reseñas de juegos hechos por distintos usuarios. Se crea la columna 'sentiment_analysis' aplicando análisis de sentimiento con NLP con la siguiente escala:
- '0' si es malo
- '1' si es neutral
- '2' si es positivo.
- De no ser posible este análisis por estar ausente la reseña escrita, debe tomar el valor de 1.
Esta nueva columna debe reemplazar la de user_reviews.review para facilitar el trabajo de los modelos de machine learning y el análisis de datos.
Para lograrlo, se utiliza la clase SentimentIntensityAnalyzer de la biblioteca NLTK (Natural Language Toolkit), que es utilizada para analizar el sentimiento de un texto.Se define una función llamada analizar_sentimiento que toma un texto como entrada y lo clasifica como positivo, negativo o neutral basado en su polaridad. Dentro de la función, se instancia un objeto SentimentIntensityAnalyzer como sia. Se calcula la polaridad del texto utilizando el método polarity_scores del objeto sia. Este método devuelve un diccionario con diferentes métricas de sentimiento, incluyendo la polaridad compuesta, que representa la polaridad general del texto en una escala de -1 (negativa) a 1 (positiva). Esta metrica es la que se empleara para determinar si el texto esta relacionado con un sentimiento positivo, negativo o neutro. 


## Praparacion de los datos para la API
Se modificaron los datasets para responder las consultas o preparar los modelos de aprendizaje automático, y de esa manera optimizar el rendimiento de la API y el entrenamiento del modelo, para lo cual se realizaron las siguientes tareas:
- Eliminacion o imputacion de nulos
- Seleccion de columnas y filtrado de registros para responder las consultas o preparar los modelos de aprendizaje automático, y de esa manera optimizar el rendimiento de la API y el entrenamiento del modelo.
- Modificacion de tipo de variables para poder manipularlas adecuadamente
- Expansion de columnas con listas o diccionarios
- Combinacion de dataframes para relacionar su informacion
- Agrupacion de registros para la obtencion de la informacion requerida
- Exportacion de dataframes a archivos .csv exceputando el dataframe para el sistema de recomendacion, que fue exportado como archivo parquet. 


## Desarrollo API y deployment: 
Se implementarán los endpoints de la API utilizando FastAPI para proporcionar acceso a los datos y a las recomendaciones de juegos. Se crean las siguientes funciones para los endpoints que se consumirán en la API:

def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}

Para realiza el deployment se emplea render. Para esto, en el siguiente link se proporciona un tutorial de como utilizarlo: https://github.com/HX-FNegrete/render-fastapi-tutorial
El link para ingresar y realizar las consultas es el siguiente: https://henry-pi-steamgames.onrender.com/docs 

## Análisis exploratorio de los datos: (Exploratory Data Analysis-EDA)

Se realiza un análisis exploratorio de los datos para investigar las relaciones entre las variables, identificar patrones interesantes, identificar outliers y nulos, tratarlos de ser necesario, mantenerlos o eliminarlos. Se generará una nubes de palabras con las reviews de los usuarios, para determinar cuales son las caracteristicas mas relevantes a tener en cuenta en el sistema de recomendacion. Esto ultimo es importante, ya que si se emplearan todas las caracteristicas para realizar el sistema de recomendacion, se obtendria un dataframe con una cantidad de columnas extremadamente grande que requeriria mucho tiempo para ser procesada y obtener la matriz de utilidad. Si bien se realiza un pequeño EDA al final del codigo, se tomo la decision de realizarlo a medida que se avanzaba en el proyecto. 

## Modelo de aprendizaje automático:

El modelo tiene una relación ítem-ítem, esto es se toma un item y en base a que tan similar esa ese ítem al resto, se recomiendan similares. Aquí el input es el id de un juego y el output es una lista de juegos recomendados, para ello se aplica la similitud del coseno. El modelo deriva en un GET/POST en la API con el siguiente formato:

def recomendacion_juego( id de producto ): Ingresando el id de producto, se recibe una lista con 5 juegos recomendados similares al ingresado.

### Cómo funciona el método de similitud del Coseno entre Ítems:
Para comenzar, se requiere una matriz que contenga todos los ítems, donde cada uno esté descrito por sus características más relevantes y las opiniones asociadas a él.

Posteriormente, se procede a calcular el coseno del ángulo entre estos vectores para obtener la similitud del coseno entre los ítems. Es importante destacar que previamente es necesario normalizar o escalar los datos, dado que este método se basa en distancias y es sensible a la magnitud de los vectores.

Una vez calculada la similitud del coseno entre todos los pares de ítems, se obtiene una medida de cuán parecidos son estos ítems en función de las calificaciones otorgadas por los usuarios. Esta información se refleja en una matriz de similitud entre ítems, donde cada celda (i, j) indica la similitud entre el ítem i y el ítem j.

Para hacer recomendaciones sobre un juego específico, se emplea la matriz de similitud entre ítems. Se identifican aquellos ítems que son más similares al juego en consideración, es decir, aquellos con una alta similitud del coseno. Estos ítems similares se sugieren al usuario, facilitando así la exploración de nuevas opciones afines a sus gustos y preferencias en el ámbito de los videojuegos.

## Video describiendo la API deployada

En el siguiente video se muestra el resultado de las consultas propuestas y del sistema de recomendacion.
https://drive.google.com/file/d/16kjaAdtVVl8Ap35u9iFj8NzQd5FMUbPV/view?usp=drivesdk
