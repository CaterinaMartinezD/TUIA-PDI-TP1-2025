import cv2
import numpy as np
import matplotlib.pyplot as plt

'''

Problema 2 - Correción de Examen Multiple Choice
El algoritmo a desarrollar debe considerar y resolver los siguientes puntos:

A. Se debe tomar únicamente como entrada la imagen de un examen y mostrar por pantalla
cuáles de las respuestas son correctas y cuáles son incorrectas.

B. Con la misma imagen de entrada, se debe validar los datos del encabezado y mostrar
por pantalla el estado de cada campo teniendo en cuentas las siguientes restricciones:
    a. Name: Debe contener al menos dos palabras y no más de 25 caracteres en total.
    b. ID: Debe contener sólo 8 caracteres en total, formando una única palabra.
    c. Code: Debe contener un único caracter.
    d. Date: Debe contener sólo 8 caracteres en total, formando una única palabra.

C. Se debe aplicar el algoritmo desarrollado, de forma cíclica, sobre el conjunto de cinco
imágenes de cada examen multiple choices resuelto (archivos multiple_choice_<id>.png)
e informar su desempeño junto con los resultados obtenidos.

D. Se debe generar una imagen de salida que informe aquellos alumnos que han aprobado
el examen (con al menos 20 respuestas correctas) y aquellos alumnos que no lo
lograron. Dicha imagen de salida debe contar con el “crop” del contenido del campo
Name del encabezado de cada examen del punto anterior, junto con algún indicador que
diferencie a aquellos que correspondan a un examen aprobado de aquellos que se
encuentren desaprobados.

'''

def leer_imagen(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    return img

def cargar_imagenes(rutas, mostrar = False):
    imagenes = []

    for ruta in rutas:
        img = leer_imagen(ruta)
        imagenes.append(img)

        if (mostrar == True):
            continue
            plt.imshow(img, cmap='gray')
            plt.title(ruta)
            plt.show()

    return imagenes

def renglones(imagenes, mostrar = True):
    resultado = []

    for idx, img in enumerate(imagenes):                                                        #Recorremos cada imagen con su indice.
        umbral = 150                                                                            
        _, binaria = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY_INV)                     # Binarizamos e invertimos la imagen con un umbral
        proyeccion = np.sum(binaria, axis=1)                                                    # Suma los píxeles por fila para obtener la proyección horizontal
        posiciones = np.where((proyeccion > 7500) & (proyeccion < 15000))[0]                    # Filtramos las posiciones donde la proyección está entre 7500 y 15000 por el indice
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                                       #convertimos a color para dibujar líneas
        renglones_detectados = []
        
        if len(posiciones) > 0:
            inicio = posiciones[0]
            for i in range(1, len(posiciones)):
                if posiciones[i] - posiciones[i - 1] < 10:                                       #Agrupa las posiciones que estan cerca (5px de diferencia)
                    continue
                else:
                    fin = posiciones[i - 1]
                    renglones_detectados.append((inicio, fin))                                  #Guarda el (inicio, fin) como coordenadas de cada renglon
                    inicio = posiciones[i]
            renglones_detectados.append((inicio, posiciones[-1]))                               #Recorre todas las posiciones

        if idx == 0:
            renglones_detectados = renglones_detectados[3:]                                     # Borra los 3 primeros para la primera imagen
        else:
            renglones_detectados = renglones_detectados[2:]                                     # Borra los 2 primeros para el resto

        for inicio, fin in renglones_detectados:                                                #dibuja las lineas horizontales en el borde superior e inferior del renglón
            cv2.line(img_color, (0, inicio), (img.shape[1], inicio), (0,255, 0), 1)             #linea de arriba
            cv2.line(img_color, (0, fin), (img.shape[1], fin), (0, 255, 0), 1)                  #linea de abajo

        #Detecta los bordes con Canny para el encabezado
        bordes = cv2.Canny(binaria, threshold1=150, threshold2=200)                             #detecta los bordes
        lines = cv2.HoughLines(bordes, rho=1, theta=np.pi/180, threshold=200)                   #detecta las lineas
        lineas_filtradas = []
        
        if lines is not None:
            for i in range(len(lines)):
                rho = lines[i][0][0]                                                            #distancia desde el origen hasta la linea
                theta = lines[i][0][1]                                                          #angulo de la línea respecto al eje X (en radianes)

                # Filtro líneas duplicadas comparando con las ya agregadas
                repetida = False
                for r, t in lineas_filtradas:
                    if abs(rho - r) < 15 and abs(theta - t) < np.pi / 180:                      #tolerancia de 15 píxeles y 2 grados aprox.
                        repetida = True                                                         #abs devuelve la diferencia absoluta entre dos valores
                        break                                                                   #np.pi es la constante π 

                if not repetida:
                    if abs(theta - np.pi/2) < np.pi / 180:                                      # Solo líneas horizontales: theta cerca de 90° (π/2 radianes)
                        lineas_filtradas.append((rho, theta))                                   #agregás si no está repetida
                        a = np.cos(theta)                                                       #convierte las coordenadas polares (ρ, θ) a coordenadas cartesianas
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))                                              #Generás dos puntos (100px hacia ambos lados) sobre la línea detectada
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(img_color, (x1, y1), (x2, y2), (255, 0, 0), 2)                 #dibujás la línea en la imagen (img_lines) en color verde (0, 255, 0) y con grosor 2

        if (mostrar == True):
            plt.figure()
            plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
            plt.title(f'Renglones detectado en imagen {idx+1}')
            plt.show()

        resultado.append(renglones_detectados)

    return resultado

def recortar(reglones, mostrar = False):
    recortes = []
    for indice, img in reglones:
        for inicio, fin in img:
            print(inicio)


rutas = ['multiple_choice_1.png','multiple_choice_2.png','multiple_choice_3.png','multiple_choice_4.png','multiple_choice_5.png']
imagenes = cargar_imagenes(rutas, mostrar=True)
renglones_img = renglones(imagenes)
recortes_img = recortar(renglones_img)












f = cv2.imread('multiple_choice_1.png', cv2.IMREAD_GRAYSCALE)
fb = cv2.medianBlur(f, 5)                                                                               # Aplicamos borrosidad
# Detectamos círculos utilizando el método de Hough cv2.HoughCircles()
#circles = cv2.HoughCircles(fb, method=cv2.HOUGH_GRADIENT, dp=1, minDist=2)                             # Valores iniciales: No detecta nada. Posible solución: disminuir umbral de votos
#circles = cv2.HoughCircles(f, method=cv2.HOUGH_GRADIENT, dp=1, minDist=2, param2=20)                   # Si bajamos mucho el umbral de votos --> Muchisimos círculos!
circles = cv2.HoughCircles(fb, method=cv2.HOUGH_GRADIENT, dp=1, minDist=3, param2=40)                   # Ajustamos umbral de votos --> Detecto solo círculos chicos
#circles = cv2.HoughCircles(fb, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param2=40, minRadius=100)  # Ajustamos radio mínimo --> Detecto solo círculos grandes

circles = np.uint16(np.around(circles))                                                                 
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
for i in circles[0,:]:
    cv2.circle(fc, (i[0],i[1]), i[2], (0,255,0), 2)   # Dibujo el círculo
    cv2.circle(fc, (i[0],i[1]), 2, (0,0,255), 2)      # Dibujo el centro del círculo

plt.figure()
plt.imshow(fc, cmap='gray')
plt.show()


#Prueba de donde saque los valores para analizar los renglones
umbral = 150
_, binaria = cv2.threshold(f, umbral, 255, cv2.THRESH_BINARY_INV)
proyeccion = np.sum(binaria, axis=1)

plt.figure()
plt.plot(proyeccion)
plt.title("Proyección Horizontal")
plt.xlabel("Fila")
plt.ylabel("Suma de píxeles")
plt.show()

# Crear un histograma de la proyección horizontal
plt.figure()
plt.hist(proyeccion, bins=50, color='gray', edgecolor='black')
plt.title("Histograma de la Proyección Horizontal")
plt.xlabel("Suma de Píxeles por Fila")
plt.ylabel("Frecuencia")
plt.show()
