import numpy as np
import cv2 as cv
import os
from pathlib import Path
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + '/haarcascade_frontalface_alt2.xml')
columns = ["class", "x", "y", "w", "h"]


def girar(img: np.ndarray):
    rows, cols, _ = img.shape
    center = (cols / 2, rows / 2)
    scale = 1
    rotated = []
    for i in range(-15, 15, 29):
        matrix = cv.getRotationMatrix2D(center, i, scale)
        rotated_image = cv.warpAffine(img, matrix, (cols, rows))
        rotated.append(rotated_image)
    return rotated


def get_face_location(image: np.ndarray):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=5)
    return faces


def delete_image(path):
    os.remove(path)


def guardar_labels(root, name, data):
    # check if file exists
    output_str = root
    csv_output_dir = Path(output_str)
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = name + ".txt"
    if (csv_output_dir / csv_filename).exists():
        return
    df = pd.DataFrame([data], columns=columns)
    df.to_csv(csv_output_dir / csv_filename, mode='a', header=False, index=False)


def generar_giradas(image, params):
    root = params["root"]
    filename = params["filename"]
    extension = params["extension"]
    rotated = girar(image)
    for i, img in enumerate(rotated):
        cv.imwrite(os.path.join(root, filename + "-" + str(i) + "." + extension), img)


def extraer_labels(image, params):
    face = params["face"]
    root = params["root"]
    filename = params["filename"]
    counter = params["counter"]
    x, y, w, h = face

    h_image, w_image, _ = image.shape
    face = [counter, x / w_image, y / h_image, w / w_image, h / h_image]
    guardar_labels(root, filename, face)


def recorrer_imagenes(image_dir, operacion):
    counter = -1
    for root, dirs, files in os.walk(image_dir):
        print(root.split("\\")[-1])
        for file in files:
            file = file.split(".")
            filename = file[0]
            extension = file[1]
            if not extension == "png" and not extension == "jpg" and not extension == "jpeg":
                continue
            file = filename + "." + extension
            path = os.path.join(root, file)
            image: np.ndarray = cv.imread(path)

            faces = get_face_location(image)
            if len(faces) != 1:
                delete_image(path)
                continue
            face = faces[0]
            operacion(
                image,
                {
                    "face": face,
                    "root": root,
                    "filename": filename,
                    "extension": extension,
                    "counter": counter
                }
            )
        counter += 1


def main():
    dir_images = input("Escribe el directorio de las imagenes: ")
    image_dir = os.path.join(BASE_DIR, dir_images)
    print("Generating rotated images...")
    recorrer_imagenes(image_dir, generar_giradas)
    print("Generating labels...")
    recorrer_imagenes(image_dir, extraer_labels)


if __name__ == '__main__':
    main()
