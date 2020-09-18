###---<ZZreadme.txt>---###

Se desarrolla aquí el análisis de los datos obtenidos , es decir, del catálogo en formato "csv" de los clumps obtenidos por la red neuronal.

El estudio tiene varias partes:

1. Fracción de galaxias clumpy según diversos parámetros (a0, cam, masa, etc.).

2. Manipulación de catálogos para extraer/añadir información.

3. Intento de clasificación en 3D de los clumps en las galaxias. Para ello, tengo un correo de Marc el 30/04 explicando como se pasa de coordenadas X,Y a coordenadas x,y en la imagen.


Componentes de esta carpeta (CLUMPS_VELA/analysis/):

- Gcat.csv : Catálogo a analizar (actualmente 30/04 , es el hecho con el ruido recortado manualmente de una zona sin estrellas de CANDELS )

- <csv> ,link to "CLUMPS_VELA/sextractor_work/sex_catalogs/csv/" : esto se hace para trabajar más comodamente con las rutas de los ficheros de los catálogos. Aparte de esto, tenemos libertad para manipular los catálogos en esta carpeta. Se ha copiado directamente de esta carpeta el catálogo importante (Gcat.csv).

- <sunrise_clump_pos> : posiciones de los clumps en el codigo SUNRISE. Esto viene explicado todo en el correo de Omri. (from "sunrise_clump_pos.tar.gz")

- <galaxy_catalogue> : todos los catalogos que se incluyen en el correo de Nir (y en la sim.) (from "galaxy_catalogue.zip")

- 3D 

- yale_next_week.txt : Extracto del correo de Mandelker que informa de todos los campos con información física necesarios de la simulación VELA.

- Mstars.txt : El catálogo que me interesa.



- ZZreadme.txt





















