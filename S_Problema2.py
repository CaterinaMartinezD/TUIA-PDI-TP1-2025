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

def renglones(imagenes, mostrar = False):
    resultado = []
    encabezados = []

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
            cv2.line(img_color, (0, inicio), (img.shape[1], inicio), (255,0, 0), 1)             #linea de arriba
            cv2.line(img_color, (0, fin), (img.shape[1], fin), (255, 0, 0), 1)                  #linea de abajo

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
        
            # Guardar el encabezado como (inicio, fin) de si hay al menos dos líneas 
            if len(lineas_filtradas) >= 2:                                                      #Verifica que haya al menos dos líneas horizontales detectadas
                ys = sorted([int(rho) for rho, _ in lineas_filtradas])                          #Guarda rho en una lista ordenada para obtener la coordenada "y" de las lineas horizontales
                encabezados.append((ys[0], ys[1]))                                              #Guardar la posición vertical (en coordenadas y) de las 2 primeras líneas horizontales

        if (mostrar == True):
            plt.figure()
            plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
            plt.title(f'Renglones detectado en imagen {idx+1}')
            plt.show()

        resultado.append(renglones_detectados)

    return resultado, encabezados

def detectar_limites_rta(imagenes, renglones_img, mostrar = False):
    resultado = []                                                                                  

    for idx_img, (img, renglones) in enumerate(zip(imagenes, renglones_img)):                   #Recorre la imagen junto los renglones detectados          
        renglones_con_limites = []
        img_color = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)                                #Convierte la imagen a BGR para dibujar los rectangulos

        for fila_i, fila_f in renglones:
            renglon = img[fila_i:fila_f, :]                                                     #Recorta el renglon de la fila_i a fila_f en todas las columnas (:)
            renglon_zeros = renglon == 0                                                        #Crea una mascara que detecta pixeles negros y devuelve (True) o (False)

            ren_col_zeros_idxs = np.argwhere(renglon_zeros.any(axis=0))                         #Se fija en que columnas tiene al menos un pixel negro y devuelve su indice
            if len(ren_col_zeros_idxs) == 0:                                                    #Si la columna tiene todos pixeles blancos (255), lo ignora
                continue

            col_i = int(ren_col_zeros_idxs[0][0])                                               #Primera columna con valores negros
            col_f = int(ren_col_zeros_idxs[-1][0])                                              #Ultima columna con valores negros

            renglones_con_limites.append((fila_i, fila_f, col_i, col_f))                        #Guarda las coordenadas en la lista

            if mostrar:
                cv2.rectangle(img_color, (col_i, fila_i), (col_f, fila_f), (0, 255, 0), 1)      #Dibuja un rectangulo de color verde (A,V,R) con grosor uno
  
        if mostrar and img_color is not None:
            plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
            plt.title(f'Imagen {idx_img+1} – Todos los renglones detectados')
            plt.show()

        resultado.append(renglones_con_limites)

    return resultado

def recorte(imagenes, renglones_img, encabezado_img, mostrar = False):
    recortes = []

    for idx_img, (img, renglones) in enumerate(zip(imagenes, renglones_img)):                    #Recorre la imagen junto con los renglones detectados
        recortes_img = []

        if encabezado_img[idx_img] is not None:                                                  
            inicio, fin = encabezado_img[idx_img]                                                #Obtiene las coordenadas del inicio y fin del encabezado
            recorte = img[inicio:fin, :]                                                         #Recorta el encabezado
            recortes_img.append(recorte)                                                         #Guarda el recorte en la lista

        for fila_i, fila_f, col_i, col_f in renglones:                                           #Recorre cada renglon teniendo en cuenta el inicio-fin vertical y horizontal
            recorte = img[fila_i:fila_f, col_i:col_f]                                            #Realiza el recorte de cada renglon de respuestas
            recortes_img.append(recorte)                                                         #Guarda el recorte en la lista

        recortes.append(recortes_img)
        if mostrar:
            for i, recorte in enumerate(recortes_img):
                plt.imshow(recorte, cmap='gray')
                plt.title(f'Imagen {idx_img + 1} – Recorte {i + 1}')
                plt.show()

    return recortes

def detectar_circulos(recortes, mostrar = False):
    resultado = []

    for idx, recortes_img in enumerate(recortes):                                               #Recorre los datos por cada indice y recorte de la lista
        recortes_img = recortes_img[1:]                                                         #Salta el primer recorte que es el encabezado
        circulos_detectados = []

        for i, recorte in enumerate(recortes_img):                                              #Recorre por cada renglon recortado
            fc = cv2.cvtColor(recorte, cv2.COLOR_GRAY2RGB)                                      #Convierte el recorte en RGB para dibujar los circulos en color
            #Detecta los circulos en los recortes
            circles = cv2.HoughCircles(recorte, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20, param2=15, minRadius=6, maxRadius=25)                                                            
            recorte_circulos = []

            if circles is not None:                                                             
                circles = np.uint16(np.around(circles))                                         #Redondea y convierte a enteros sin signo a cada valor obtenido
                for c in circles[0, :]:                                                         #Recorre cada circulo detectado
                    recorte_circulos.append((c[0], c[1], c[2]))                                 #Guarda las coordenadas (x, y, radio) en la lista
                    cv2.circle(fc, (c[0], c[1]), c[2], (0, 255, 0), 2)                          #Dibuja el circulo en verde con grosor 2
                    cv2.circle(fc, (c[0], c[1]), 2, (0, 0, 255), 2)                             #Dibuja el centro del circulo en rojo con grosor 2

            circulos_detectados.append(recorte_circulos)

            if mostrar:
                plt.imshow(cv2.cvtColor(fc, cv2.COLOR_BGR2RGB))
                plt.title(f'Imagen {idx + 1} – Recorte {i + 1}')
                plt.show()

        resultado.append(circulos_detectados)

    return resultado

def detectar_rta_respondida(circulos, recortes):
    letras = ['A', 'B', 'C', 'D', 'E']                                                           #Realizamos una lista de las letras que aparecen
    resultado = []

    for idx_imagen, (circulos_img, recortes_img) in enumerate(zip(circulos, recortes)):         #Recorre la imagen respecto a los circulos y recortes
        recortes_preguntas = recortes_img[1:]                                                   #Salta el primer recorte que contiene el encabezado
        rtas_imagen = []

        #Recorre cada recorte junto los circulos detectados
        for idx_recorte, (recorte, lista_circulos) in enumerate(zip(recortes_preguntas, circulos_img)):
            umbral = 150                                                                        
            _, binaria = cv2.threshold(recorte, umbral, 255, cv2.THRESH_BINARY_INV)             #Convierte el recorte en una imagen binaria invertida
            pixeles_por_circulo = []
            lista_circulos_ordenados = sorted(lista_circulos, key=lambda c: c[0])               #Ordenamos los circulos horizontalmente con la coordenada "x"
            
            for (x, y, r) in lista_circulos_ordenados:                                          #Recorre cada circulo por su posicion
                x, y, r = int(x), int(y), int(r)                                                #convierte las coordenadas a enteros

                #Forma un cuadrado alrededor del circulo
                x1 = max(0, x - r)
                x2 = min(binaria.shape[1], x + r)
                y1 = max(0, y - r)
                y2 = min(binaria.shape[0], y + r)  
                celda = binaria[y1:y2, x1:x2]                                                   #Recorta esa seccion de la imagen binaria
                
                mask = np.zeros_like(celda, dtype=np.uint8)                                     #Genera una imagen del mismo tamaño que el recorte con valores ceros
                cv2.circle(mask, (x - x1, y - y1), r, 255, thickness=-1)                        #Dibuja un circulo blanco en la mascara
                pixeles_activos = cv2.countNonZero(cv2.bitwise_and(celda, mask))                #Cuenta cuantos pixeles blancos hay dentro de la mascara
                pixeles_por_circulo.append(pixeles_activos)                                     #Guarda la cantidad de pixeles detectados

            cantidad_marcados = 0                                                               #Contador de circulos marcados
            indice_marcado = -1                                                                 #Guarda el indice si hay solo uno
            for i in range(len(pixeles_por_circulo)):                                           #Recorre los valores de pixeles respecto a los contados
                if pixeles_por_circulo[i] > 250:                                                #Si un circulo tiene mas de 250px blancos se considera marcada
                    cantidad_marcados += 1
                    if cantidad_marcados == 1:
                        indice_marcado = i

            if cantidad_marcados == 0:                                                          #Si no se marco ninguna se considera incorrecta (sin responder)
                rtas_imagen.append("No respondida")
            elif cantidad_marcados == 1:                                                        #Si marco una, se le asigna su letra respecto a su pocicion
                rtas_imagen.append(letras[indice_marcado])
            else:
                rtas_imagen.append("Múltiples")                                                 #Sino se considera incorrecta (multiples opciones marcadas)

        resultado.append(rtas_imagen)
    return resultado

def corregir_y_mostrar(letras_img, imagenes, renglones_img, mostrar=False):
    #Se genera una lista con las respuestas correctas
    respuestas = ['A', 'A', 'B', 'A', 'D', 'B', 'B', 'C', 'B', 'A', 'D', 'A', 'C', 'C', 'D', 'B', 'A', 'C', 'C', 'D', 'B', 'A', 'C', 'C', 'C']
    resultado = []

    for idx_img, letras in enumerate(letras_img):                                                #Recorre indice y letra respecto a las rta del alumno
        img_resultado = []
        for idx, letra in enumerate(letras):                                                     #Recorre cada letra respecto a su posicion
            #Compara con las respuestas correctas
            if letra == respuestas[idx]:                                                         
                img_resultado.append("OK")                                                       #Si es correcta la respuesta guarda "OK" en la lista                      
            else:
                img_resultado.append("MAL")                                                      #Si es incorrecta la respuesta guarda "MAL" en la lista  

        resultado.append(img_resultado)                                                          #Guarda los resultados de esa imagen

    for idx_img, (img, renglones, correcciones) in enumerate(zip(imagenes, renglones_img, resultado)):
        img_copia = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)                                 #Convierte la imagen a RGB para dibujar la rta en color

        for (fila_i, fila_f, col_i, col_f), correccion in zip(renglones, correcciones):          #Recorre cada renglon respecto a sus correcciones
            x_texto = col_f + 15
            y_texto = fila_f
            cv2.putText(img_copia, correccion, (x_texto, y_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

        if mostrar:
            plt.imshow(img_copia)
            plt.title(f"Imagen {idx_img + 1} – Correcciones")
            plt.show()

    return resultado


#------------------------------------------------------------------------------------------------------------------------------------

rutas = ['multiple_choice_1.png','multiple_choice_2.png','multiple_choice_3.png','multiple_choice_4.png','multiple_choice_5.png'] 
imagenes = cargar_imagenes(rutas, mostrar = False)
renglones_img, encabezado = renglones(imagenes, mostrar = False)
renglones_con_limites = detectar_limites_rta(imagenes, renglones_img, mostrar = False)
recortes_img = recorte(imagenes, renglones_con_limites, encabezado, mostrar = False)
detectar_circulos_img = detectar_circulos(recortes_img, mostrar = False)
rta_alumno = detectar_rta_respondida(detectar_circulos_img, recortes_img)
correccion_respuestas_img = corregir_y_mostrar(rta_alumno, imagenes, renglones_con_limites, mostrar = False)



'''

#Para ver como se detectan las letras
for idx_img, rtas in enumerate(detectar_letras_img):
    print(f"\nResultados Imagen {idx_img+1}")
    for q_num, letra in enumerate(rtas, start=1):
        print(f"Pregunta {q_num:2d}: {letra}")

        
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

'''