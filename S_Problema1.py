import cv2
import numpy as np
import matplotlib.pyplot as plt

'''

Problema n° 01 - Ecualización del histograma

Desarrolle una función para implementar la ecualización local del histograma, que reciba como parámetros de entrada la imagen a 
procesar, y el tamaño de la ventana de procesamiento (MxN). Utilice dicha función para analizar la imagen que se muestra en la 
Figura 1 e informe cuáles son los detalles escondidos en las diferentes zonas de la misma. 

'''

def histograma(img, M, N):
    bordes = cv2.copyMakeBorder(img, M//2, M//2, N//2, N//2, cv2.BORDER_REPLICATE)   #Genera una imagen con bordes para poder aplicar la ventana centrada en los bordes.
    img_t = np.zeros_like(img)                                                       #Crea una imagen vacía del mismo tamaño y tipo de datos que la original. (valor = 0)
    filas, columnas = img_t.shape                                                    #Se obtienen las dimensiones de la imagen.

    for x in range(filas):                                                          
        for y in range(columnas):                                                    #Recorre cada pixel por cada fila y columna.
            ventana = bordes[x:x + M, y: y + N]                                      #Obtiene una ventana de tamaño MxN centrada en el píxel actual.
            ventana_eq = cv2.equalizeHist(ventana)                                   #Realiza la ecualización del histograma en la ventana.
            img_t[x, y] = ventana_eq[M//2, N//2]                                     #Coloca el resultado obtenido en su respectiva posición.

    return img_t


img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
img_t1 = histograma(img, 3, 3)
img_t2 = histograma(img, 25, 25)
img_t3 = histograma(img, 51, 51)

plt.figure(figsize=(12, 5))
plt.title("Imagen con detalles escondidos"), plt.axis('off')
plt.subplot(131)
plt.imshow(img_t1, cmap='gray'), plt.title("Kernel = 3x3")
plt.subplot(132)
plt.imshow(img_t2, cmap='gray'), plt.title("Kernel = 25x25")
plt.subplot(133)
plt.imshow(img_t3, cmap='gray'), plt.title("Kernel = 51x51")
plt.show()
