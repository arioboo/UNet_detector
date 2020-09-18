###----------------<>---------------###


La imagen VELA07_a0.410_sunrise_cam17_ACS-F606W_SB00.fits y VELA07_a0.410_sunrise_cam17_ACS-F606W_SB00-pred.fits , es un 
buen candidato a analizar con el ds9 puesto que tiene muchos clumps. Podemos usarla para testear la veracidad del ds9.

#Pasos:

1. Cargar los catálogos sex_cat_VELA07_F606W_realHSTlikenoise.csv y sex_cat_VELA07_F606W.csv con topcat.
2. Modificar los campos con las fórmulas de query de "NUMBER"(numero de clumps). Así se puede ver imágenes con muchos clumps (i.e.,la dicha anteriormente).
3. Abrir en el ds9 las imagenes REAL,PREDICCIÓN,SEGMENTACIÓN del mismo tipo de imagen. Como opcion, puede añadirse la imagen de ruido(NO HACER). Cargar en topcat los campos de la imagen en cuestión, seleccionando "X","Y" e "imagen" 

4. Repetir los pasos 1-3 para los distintos tipos de ruido que se deseen probar.

