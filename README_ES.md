# Sistema de Recomendación de Películas

Un sistema de recomendación de películas construido con tres enfoques independientes y
comparables entre sí: filtrado basado en contenido, filtrado colaborativo item based y
factorización de matrices (SVD). Los tres se evalúan sobre los mismos datos reservados y se
comparan directamente, con sus fallos y su comportamiento en frío documentados en lugar de
ocultados.

- **Problema:** Recomendar películas a partir de tres señales genuinamente distintas (metadata,
  comportamiento de coincidencia entre usuarios, y factores latentes aprendidos), cubriendo
  tanto el caso de usuario o película ya existente como el caso de arranque en frío donde
  ninguno de los dos existe todavía
- **Resultado:** El colaborativo obtiene Precision@5 de 42,41% y Recall@5 de 15,16% frente al
  5,93%/1,87% del basado en contenido y al 6,05%/1,57% de SVD, una diferencia de entre 7 y 8
  veces que el notebook explica en vez de limitarse a reportar
- **Valor:** Un pipeline de recomendación completo y auditado, no un único modelo entrenado: bugs
  reales encontrados y corregidos con evidencia en cada etapa, arranque en frío resuelto de forma
  explícita para dos de los tres modelos, y un relato honesto de qué pueden y qué no pueden
  demostrar las métricas offline con un dataset de este tamaño

> [View this project in English](README.md)

---

## Tabla de contenidos

1. [Definición del problema](#definición-del-problema)
2. [Dataset](#dataset)
3. [Limpieza de datos](#limpieza-de-datos)
4. [Análisis exploratorio de datos](#análisis-exploratorio-de-datos)
5. [Filtrado basado en contenido](#filtrado-basado-en-contenido)
6. [Filtrado colaborativo](#filtrado-colaborativo)
7. [Factorización de matrices (SVD)](#factorización-de-matrices-svd)
8. [Evaluación](#evaluación)
9. [Cómo decide cada modelo](#cómo-decide-cada-modelo)
10. [Problemas encontrados](#problemas-encontrados)
11. [Conclusiones de negocio](#conclusiones-de-negocio)
12. [Para quién es útil](#para-quién-es-útil)
13. [Valor del proyecto](#valor-del-proyecto)
14. [Posibles mejoras](#posibles-mejoras)
15. [Requisitos](#requisitos)

---

## Definición del problema

Recomendar películas a un usuario a partir de uno de estos dos tipos de entrada: una película
que ya le gusta ("más como esta"), o su propio historial de valoraciones ("para ti"). Se
construyen tres modelos para responder a esto, cada uno apoyado en una señal distinta:

| Modelo | Señal que usa |
|---|---|
| Basado en contenido | Género, director, reparto, keywords, productora, saga |
| Colaborativo (item based) | Qué usuarios valoraron qué películas, y cómo |
| SVD | Factores latentes aprendidos a partir de esas mismas valoraciones |

Un recomendador que solo funciona con usuarios y títulos ya bien establecidos tiene un valor
real limitado, así que el proyecto también cubre de forma explícita el caso de arranque en frío:
un usuario sin historial de valoraciones, y una película sin interacciones.

---

## Dataset

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (Kaggle,
compilado a partir de TMDB y MovieLens), descargado con `kagglehub`.

| Fichero | Contenido |
|---|---|
| `movies_metadata.csv` | ~45.000 películas: título, géneros, presupuesto, recaudación, duración, idioma, fecha de estreno, productoras/países, votos, sinopsis |
| `credits.csv` | Reparto y equipo por película (codificado en JSON), usado para director y actores principales |
| `keywords.csv` | Palabras clave de trama por película (codificado en JSON) |
| `links_small.csv` | Mapeo entre `movieId` de MovieLens, `imdbId` de IMDb y `tmdbId` de TMDB |
| `ratings_small.csv` | Valoraciones de usuarios (`userId`, `movieId`, `rating`, `timestamp`) |

El `ratings.csv` completo (unos 26 millones de valoraciones) no cabe en memoria en local, así
que este proyecto usa las versiones reducidas de principio a fin. Esa decisión fija el alcance
del proyecto: tras la limpieza y deduplicación, el catálogo queda en **7.818 películas**, y
`ratings_small.csv` cubre **671 usuarios** con unas **100.000 valoraciones** en total (entre 16
y 1.391 por usuario).

---

## Limpieza de datos

Dos uniones independientes (metadata con reparto/keywords, y metadata con links/ratings) se
combinan en una tabla a nivel de valoración, limpiada antes de construir ningún modelo:

- Columnas codificadas en JSON (`genres`, `cast`, `crew`, `keywords`, `production_companies`,
  `belongs_to_collection`) parseadas con `ast.literal_eval` a estructuras de Python utilizables.
- Corrección de tipos: `budget` y `revenue` convertidos a numérico, `release_date` a fecha,
  `imdb_id` sin el prefijo `tt`.
- Duplicados y filas casi vacías eliminadas; una película sin ninguna columna relevante de
  contenido se excluye en vez de mantenerse con texto de relleno.
- `production_countries` y `spoken_languages` eliminadas pronto: ambas muy concentradas (cerca
  del 70% en EEUU, cerca del 95% en inglés) y con poca señal discriminativa, y quitarlas antes
  del filtro de valores en blanco significa que un hueco en cualquiera de las dos ya no le cuesta
  su sitio en el dataset a una película por lo demás completa.

Durante esta fase se encontraron y corrigieron varios bugs reales: un `.median` sin paréntesis
corrompiendo columnas numéricas en silencio, `budget` sin convertir a numérico nunca,
`belongs_to_collection` volviendo a nulo tras el parseo de JSON, y `crew` uniéndose carácter a
carácter en vez de por elementos de la lista.

---

## Análisis exploratorio de datos

Once preguntas, cada una con una hipótesis planteada, un test estadístico donde aplica, y un
tamaño del efecto, no solo un p valor. El detalle completo y el código están en el notebook;
hallazgos principales:

- **Géneros.** Drama (4.038) y Comedy (2.889) dominan; Foreign (34) y TV Movie (24) son casi
  inexistentes, demasiado pocas para sostener comparaciones significativas por género.
- **Presupuesto.** Muy asimétrico a la derecha (mediana $20M, media $33,65M); el 43,4% del
  catálogo no tiene presupuesto reportado.
- **Presupuesto vs. recaudación.** Correlación positiva, Pearson r = 0,71 (r² = 0,51) excluyendo
  ceros.
- **vote_average vs. vote_count.** Solo una relación débil (r² ≈ 0,04 a 0,05); la popularidad de
  una película dice muy poco sobre lo bien valorada que está en media.
- **Idioma.** El inglés domina con un 88,1%, impulsado por el volumen de producción de Hollywood
  y un sesgo de muestreo hacia audiencias occidentales en el propio dataset, no por número de
  hablantes (el mandarín tiene más).
- **Nivel de presupuesto vs. nota.** Las películas de bajo presupuesto puntúan más alto en el
  64,1% de las comparaciones por pares (Mann-Whitney, efecto pequeño a moderado).
- **Idioma × década.** Estadísticamente significativo (p = 3,5e-32) pero de efecto prácticamente
  nulo (V de Cramér = 0,074), un recordatorio de que significancia y tamaño del efecto responden
  preguntas distintas.
- **vote_count por década.** El hallazgo más fuerte del EDA (Kruskal-Wallis, η² = 0,218): las
  décadas más recientes tienen muchos más votos, un artefacto del crecimiento de la plataforma,
  no de la calidad de las películas.

---

## Filtrado basado en contenido

Los campos `genres`, `director`, `cast`, `keywords`, `production_companies` y
`belongs_to_collection` de cada película se combinan en una única cadena de texto normalizada
(`text`), en minúsculas y sin espacios internos, para que un nombre completo cuente como un solo
token en vez de dividirse en palabras comunes. Pertenecer a la misma saga resultó ser una de las
señales más fuertes disponibles, más que el género para ese caso concreto, así que las películas
sin saga usan su propio título como sustituto.

Se construyen y comparan `CountVectorizer` y `TfidfVectorizer`. En películas ya presentes en el
catálogo coinciden de cerca en la señal fuerte (Toy Story devuelve Toy Story 2/3 con los dos) y
divergen en la cola: al consultar `Sabrina (1995)`, `CountVectorizer` devuelve otras comedias
románticas de época clásica, mientras que `TfidfVectorizer` devuelve la propia filmografía de
Sydney Pollack, el director de esa película en concreto, porque los tokens raros pesan mucho más
que las palabras de género comunes.

El **arranque en frío** para una película que no existe en absoluto en el catálogo funciona:
`vectorizer.transform()` reutiliza el vocabulario ya ajustado sin reentrenar nada, y los tokens
desconocidos simplemente se ignoran. Probado con *Dune: Part Two*, estrenada después de
construirse este dataset. Un parámetro `threshold` (por defecto `0,1` para la función de
arranque en frío, `0,0` para la de catálogo existente) protege contra el caso en que un texto
escrito a mano no comparte ningún vocabulario con el corpus: sin él, una similitud de exactamente
`0,0` contra todas las películas sigue siendo un orden de clasificación válido, así que la
función devolvería en silencio lo que sea que quedara al final tras el desempate, disfrazado de
recomendación real.

---

## Filtrado colaborativo

Se construye una matriz usuario-película (`userId` × `movieId`, valores son las notas, `0` para
lo no valorado) a partir de `ratings_small.csv`, y la similitud entre películas se calcula como
similitud coseno entre columnas de esa matriz, sin ninguna metadata de por medio. Rellenar con
ceros es seguro aquí porque las notas nunca llegan a 0 en la escala de 0,5 a 5,0 de este dataset,
y porque un cero no aporta nada ni al producto punto ni a la norma del vector, así que la
comparación se calcula en la práctica solo sobre los usuarios que valoraron ambas películas.

Las películas con menos de 5 valoraciones se filtran antes de calcular la similitud: con una
sola valoración, la similitud de esa película con cualquier otra que ese mismo usuario también
valorara sale en torno a 1,0, sea cual sea la nota real, una coincidencia perfecta falsa, no una
señal débil. Ese corte reduce el catálogo de 7.818 películas a **3.357 (43%)**, la mayor
limitación estructural de este modelo.

El **arranque en frío de usuario nuevo está resuelto**: una lista corta de títulos que el
usuario dice que le gustan se trata como pseudo valoraciones con el mismo peso, reutilizando la
misma suma ponderada de filas de similitud que se usa para las valoraciones reales de un usuario
existente. El **arranque en frío de película nueva no está resuelto**, y no puede resolverse
desde dentro de este modelo: un título solo se gana un sitio en la matriz de similitud una vez
que gente real la ha valorado suficientes veces. Ese hueco queda documentado como una limitación
declarada, no parcheado con un fallback al basado en contenido; los dos modelos se mantienen
independientes a propósito, para que el trade off entre ambos siga siendo visible.

---

## Factorización de matrices (SVD)

Construido con `SVD` de `surprise` como tercer modelo, independiente de los otros dos: cada
usuario y cada película obtienen un vector de factores latentes, y una nota predicha es
`media_global + sesgo_usuario + sesgo_pelicula + producto_punto(vector_usuario,
vector_pelicula)`. A diferencia de una descomposición densa genérica, `surprise` entrena solo
sobre las valoraciones que existen de verdad, nunca sobre los ceros de los que depende el modelo
colaborativo como relleno.

El RMSE sobre valoraciones reservadas es **0,8735**, en el rango habitual para SVD sobre datos
tipo MovieLens (el sistema base del Netflix Prize puntuaba 0,95, el equipo ganador 0,857 tras
años de ajuste). Ese número por sí solo no dice nada sobre personalización: al probar con dos
usuarios muy distintos (uno de gustos dispersos y otro mainstream), ambos convergían hacia el
mismo pequeño grupo de clásicos de consenso antes de fijar `random_state`, y el solapamiento se
redujo pero no desapareció una vez fijado. Con tan solo 16 valoraciones para algunos usuarios, el
término personalizado de la fórmula tiene poco de lo que aprender, así que la popularidad
general de una película domina la predicción más que en cualquiera de los dos modelos
anteriores.

El arranque en frío de usuario nuevo es un hueco más difícil aquí que en el colaborativo:
`svd.predict()` para un `userId` nunca visto cae a la media global de valoraciones para
cualquier película, una predicción que parece válida pero no lleva ninguna personalización, y
cerrar ese hueco necesitaría un reentrenamiento completo o un paso de fold-in que este proyecto
no implementa.

---

## Evaluación

Los tres modelos se comparan con Precision@5 y Recall@5 sobre exactamente los mismos datos
reservados por usuario: el 80% de las valoraciones de cada usuario se usan como entrada, el
resto es el conjunto de test, y una película cuenta como relevante si se valoró con `3` o más,
el punto medio de la escala 0,5 a 5,0.

| Modelo | Precision@5 | Recall@5 |
|---|---|---|
| Colaborativo (item based) | 42,41% | 15,16% |
| SVD | 6,05% | 1,57% |
| Basado en contenido | 5,93% | 1,87% |

El colaborativo gana por entre 7 y 8 veces en las dos métricas, y es lo esperable, no una
sorpresa: la señal de relevancia aquí es una nota reservada, justo la tarea para la que está
construido el colaborativo y justo la tarea a la que el basado en contenido nunca tuvo acceso.
La diferencia dice qué modelo predice mejor *notas ya reservadas*, no cuál recomendación
elegiría de verdad un usuario real, una pregunta distinta que necesitaría usuarios reales viendo
recomendaciones reales para responderse. Tampoco significa que las recomendaciones del basado en
contenido estén mal: que un usuario nunca haya valorado una película no es evidencia de que le
fuera a disgustar, solo de que probablemente nunca la ha visto, así que precision@k
*subestima sistemáticamente* a cualquier modelo bueno enseñando algo que el usuario todavía no
ha encontrado, que es justo el comportamiento que necesita un recomendador útil.

---

## Cómo decide cada modelo

| | Basado en contenido | Colaborativo (item based) | SVD |
|---|---|---|---|
| Entrada que necesita | La propia metadata de una película | Valoraciones de muchos usuarios | Valoraciones de muchos usuarios |
| Cálculo central | Similitud coseno entre vectores TF-IDF/conteo | Similitud coseno entre columnas de notas | `producto_punto(vector_usuario, vector_pelicula)` más sesgos |
| Qué significa "parecido" | Mismo tipo de objeto (género, reparto, saga) | Vista por la misma gente | Cercana en un espacio de factores aprendido, sin etiqueta |
| Película nueva | Funciona al momento desde una descripción | Imposible hasta acumular valoraciones | Imposible hasta acumular valoraciones |
| Usuario nuevo | Funciona desde un título que le guste | Funciona desde varios títulos que le gusten | Cae a la media global, sin personalización |
| Fallo típico | Recomienda más de lo mismo, nunca sorprende | Coincidencias perfectas falsas con datos muy escasos | Converge hacia títulos genéricamente populares sin importar quién pregunte |

Los tres se mantienen independientes a propósito, ninguno cae de vuelta a otro cuando no puede
responder, para que el trade off entre ellos siga siendo visible en vez de quedar oculto detrás
de una única lista mezclada.

---

## Problemas encontrados

Una muestra representativa de bugs reales, encontrados y corregidos con evidencia mientras se
construía este proyecto, cada uno confirmado reproduciéndolo antes de arreglarlo:

- **Títulos duplicados rompían las búsquedas.** Remakes y reboots que comparten título (Sabrina
  1954/1995, Ghostbusters 1984/2016) hacían que una búsqueda de título a id devolviera más de
  una fila, lanzando `KeyError`/`ValueError`. Arreglado añadiendo el año de estreno a cada título
  duplicado.
- **Un fallo silencioso peor que un error.** Una consulta de arranque en frío sin ningún
  solapamiento de vocabulario producía una similitud coseno de exactamente `0,0` contra todas las
  películas, que sigue siendo un orden de clasificación válido, así que la función devolvía lo
  que quedara al final tras el desempate, con pinta de recomendación real. Arreglado con un
  parámetro `threshold` que devuelve `'No movies found'` en su lugar.
- **`SVD()` sin `random_state` no era reproducible.** Dos usuarios muy distintos compartían
  varias de sus recomendaciones principales entre ajustes independientes del modelo antes de
  detectar esto, y el solapamiento cambiaba en cada reentrenamiento, sin relación con ninguna
  señal real de los datos.
- **Un split de train/test coordinado importaba más de lo que parecía.** SVD se evaluó primero
  contra un split calculado de forma independiente al usado para entrenarlo, lo que significaba
  que podía llevarse el mérito de una nota que ya había visto durante el entrenamiento. Arreglado
  derivando ambos del mismo split.
- **`.loc` frente a `[]` simple en un DataFrame.** Seleccionar filas con una lista de etiquetas
  de fila mediante `[]` simple, pandas lo interpreta como una petición de columnas, no de filas,
  un error que apareció más de una vez en los dos modelos y siempre falló lo bastante fuerte
  como para detectarse, pero fácilmente podría no haberlo hecho.

---

## Conclusiones de negocio

- **Dos de los tres modelos resuelven su caso de arranque en frío más difícil; ninguno los
  resuelve todos.** El basado en contenido no tiene problema de arranque en frío en absoluto, ya
  que nunca necesita datos de interacción. Colaborativo y SVD resuelven ambos el caso de usuario
  nuevo, pero ninguno puede recomendar una película que todavía no ha acumulado valoraciones
  reales, un hueco estructural ligado a cómo está construido cada modelo, no algo que arregle un
  cambio de parámetro.
- **Más precisión en esta métrica offline no es lo mismo que un producto mejor.** La ventaja de
  entre 7 y 8 veces del colaborativo en Precision@5/Recall@5 refleja que la métrica mide
  exactamente la tarea para la que fue construido. Una decisión de producto entre "más como
  esta" y "para ti" debería guiarse por lo que el usuario intenta hacer, no por qué modelo gana
  un número offline calculado sobre una pregunta distinta.
- **El 43% de cobertura del catálogo es el coste real de la fortaleza del modelo colaborativo.**
  El mismo corte de 5 valoraciones que mantiene fiables sus similitudes también significa que el
  57% del catálogo no puede recomendarse, ni consultarse, a través de él. Cualquier producto
  construido sobre este modelo necesita un plan para esa mayoría del catálogo, ya sea filtrado
  basado en contenido, un fallback de popularidad, o ambos.
- **Los números de evaluación aquí responden a una pregunta más estrecha que "cuál modelo es
  mejor".** Dicen qué modelo predice mejor una nota que ya ocurrió, no qué recomendación elegiría
  de verdad un usuario real. Cerrar ese hueco de verdad necesita lo mismo que cualquier
  recomendador en producción necesita antes de poder afirmar su calidad con confianza: usuarios
  reales, viendo recomendaciones reales, con su siguiente acción observada después.

---

## Para quién es útil

**Personas.** Una lista "para ti" construida a partir de tu propio historial de valoraciones
(colaborativo o SVD), o una lista "más como esta" a partir de una sola película que ya te gustó
(basado en contenido), incluyendo títulos que ni siquiera están en el dataset si puedes
describirlos, útil para descubrir más allá de lo que una plataforma ya sabe que has visto.

**Empresas.** Una plataforma de streaming o alquiler obtiene tres estrategias desplegables en
vez de una: filtrado colaborativo para el 43% del catálogo con suficiente dato de interacción,
filtrado basado en contenido para todo lo demás, incluidos estrenos nuevos desde el momento en
que existe su metadata, y un relato documentado y cuantificado del trade off entre ambos para
decidir dónde merece la pena el coste de ingeniería de cada uno. La misma arquitectura (modelos
independientes, arranque en frío gestionado de forma explícita, evaluación offline equiparada)
generaliza más allá de las películas a cualquier catálogo con metadata rica de producto y datos
de interacción de usuario: libros, música, productos de ecommerce, artículos.

---

## Valor del proyecto

Esto cubre el ciclo completo de un proyecto de recomendación de principio a fin, no un único
modelo entrenado y dejado en un notebook: limpieza de datos y EDA fundamentado en tests de
hipótesis en vez de en impresión visual, tres enfoques de recomendación distintos con sus fallos
documentados en vez de ocultos, evaluación formal con métricas de recuperación estándar junto
con un relato explícito de qué pueden y qué no pueden zanjar esas métricas con un dataset de este
tamaño, y un relato desde cero de cada bug real encontrado por el camino, no solo del código que
funcionó. Esa última parte es el diferencial: buena parte de lo que hace fiable a este proyecto
es visible en el notebook precisamente porque los errores se dejaron en él, no se editaron fuera.

---

## Posibles mejoras

**Modelado**

- **Autoencoders.** Un autoencoder profundo entrenado para reconstruir un vector de
  valoraciones puede capturar patrones de interacción no lineales que la similitud coseno no
  puede, y tiende a superar al filtrado colaborativo clásico una vez hay datos suficientes para
  entrenarlo.
- **Embeddings de texto para el contenido.** `text` aquí es una bolsa de tokens de metadata
  estructurada, con `overview` excluida a propósito. Embeddings de frase de un modelo de
  lenguaje sobre la sinopsis real capturarían similitud temática que ninguna lista de keywords
  captura.
- **LLMs como componente.** Un LLM podría reordenar o explicar una lista corta ya producida por
  estos modelos, o actuar como juez en lugar de las métricas offline que este proyecto no pudo
  usar de forma justa entre modelos.
- **Un híbrido de verdad mezclado.** Este proyecto mantiene los tres modelos independientes a
  propósito para que el trade off siga siendo visible; un sistema en producción podría en cambio
  mezclarlos, cambiando esa visibilidad por una única lista de mejor rendimiento.

**Datos**

- **El dataset completo.** Con 25 millones de valoraciones en vez de 100.000, el mínimo de 5
  valoraciones exigido para el colaborativo probablemente podría bajarse bastante, reduciendo o
  cerrando el hueco de cobertura del 43%.
- **Señal temporal.** `timestamp` se descartó pronto y nunca se usó; la recencia es una señal
  real que un modelo estático como este ignora por completo.
- **Evaluación online.** Lo único que ninguna métrica offline aquí puede sustituir: usuarios
  reales, viendo recomendaciones reales, con su siguiente acción observada.

---

## Requisitos

```
pandas==3.0.5
numpy==2.5.1
matplotlib==3.11.1
seaborn==0.13.2
scipy==1.18.0
scikit-learn==1.9.0
scikit-posthocs==0.14.0
scikit-surprise==1.1.5
kagglehub==1.0.2
```

Instalar todas las dependencias:

```bash
pip install -r requirements.txt
```

Este proyecto se construyó con Python 3.12.

---

*Fuente del dataset: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset*
