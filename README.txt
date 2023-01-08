https://github.com/LuisTorro/Proyecto_ML

OBJETIVO DEL PROYECTO: 
El proyecto tiene como finalidad comprobar si es posible automatizar la clasificación 
de libros en géneros únicamente por su portada y/o su título

DATOS: 

divididos en train y test imágenes y titulo del libro:

30 classes

Training - 51,300 Total

Test - 5,700 Total

|Label|Category Name|Training Size|Test Size|
|---|---|---|---|
|0|Arts & Photography|1,710|190|
|1|Biographies & Memoirs|1,710|190|
|2|Business & Money|1,710|190|
|3|Calendars|1,710|190|
|4|Children's Books|1,710|190|
|5|Comics & Graphic Novels|1,710|190|
|6|Computers & Technology|1,710|190|
|7|Cookbooks, Food & Wine|1,710|190|
|8|Crafts, Hobbies & Home|1,710|190|
|9|Christian Books & Bibles|1,710|190|
|10|Engineering & Transportation|1,710|190|
|11|Health, Fitness & Dieting|1,710|190|
|12|History|1,710|190|
|13|Humor & Entertainment|1,710|190|
|14|Law|1,710|190|
|15|Literature & Fiction|1,710|190|
|16|Medical Books|1,710|190|
|17|Mystery, Thriller & Suspense|1,710|190|
|18|Parenting & Relationships|1,710|190|
|19|Politics & Social Sciences|1,710|190|
|20|Reference|1,710|190|
|21|Religion & Spirituality|1,710|190|
|22|Romance|1,710|190|
|23|Science & Math|1,710|190|
|24|Science Fiction & Fantasy|1,710|190|
|25|Self-Help|1,710|190|
|26|Sports & Outdoors|1,710|190|
|27|Teen & Young Adult|1,710|190|
|28|Test Preparation|1,710|190|
|29|Travel|1,710|190|

MODELOS:

clasificación de imágenes y NLP con Tfid vectorizer

RESULTADOS:

clasificación de imágenes: 25,43% acc | proy_ML_ImgClass | MODELO: my_model.hdf5
NLP con Tfid vectorizer: 58.05% acc | proy_ML_NLP | MODELO: my_model_LR.pkl
