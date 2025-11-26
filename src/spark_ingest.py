import os
import cv2
import base64
import pandas as pd
import numpy as np

# Definir la ruta del directorio de dataset
dataset_dir = "dataset"
output_dir = "warehouse"

# Asegúrate de que el directorio existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Directorio de dataset: {os.path.abspath(dataset_dir)}")
print(f"¿Existen las carpetas de dataset?: {os.path.exists(dataset_dir)}")
print(f"Contenido de dataset: {os.listdir(dataset_dir)}")

# Listas para almacenar las imágenes y las etiquetas
images = []
labels = []

# Recorrer cada subcarpeta (cada persona) dentro de dataset/
for person_folder in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_folder)

    # Asegurarse de que estamos trabajando con un directorio
    if os.path.isdir(person_path):
        print(f"Procesando imágenes de: {person_folder}")

        # Recorrer las imágenes dentro de cada subcarpeta
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            print(f"Procesando archivo: {image_file}")

            # Verificar que sea una imagen (.jpg o .png)
            if image_file.endswith(".jpg") or image_file.endswith(".png"):
                print(f"Cargando imagen: {image_path}")
                img = cv2.imread(image_path)
                
                if img is not None:
                    #Comando para cambiar la imagen en blanco y negro
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))  # Redimensionar a 200x200

                        # Convertir la imagen (que es un array de bytes) a un string de Base64
                        _, img_encoded = cv2.imencode('.png', face_roi)
                        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                        # Guardar la imagen codificada en Base64 y la etiqueta (nombre de la persona)
                        images.append(img_base64)
                        labels.append(person_folder)

                else:
                    print(f"No se pudo cargar la imagen: {image_path}")
            else:
                print(f"El archivo {image_file} no es una imagen válida (.jpg/.png)")

# Verificar si se han encontrado imágenes y etiquetas
if len(images) == 0:
    print("No se encontraron imágenes válidas en el dataset.")
else:
    # Convertir las listas a un DataFrame de pandas
    df = pd.DataFrame({'image': images, 'label': labels})

    # Guardar los datos en un archivo Parquet (o CSV si prefieres)
    parquet_file = os.path.join(output_dir, 'faces.parquet')

    # Usar pandas para guardar en formato Parquet
    df.to_parquet(parquet_file, engine='pyarrow')

    print(f"Archivo Parquet guardado en: {parquet_file}")
