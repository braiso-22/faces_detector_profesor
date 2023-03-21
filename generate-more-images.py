import random

import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from numpy import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def ruido(img: np.ndarray, cantidad: int):
    ruido_normal: np.ndarray = random.normal(10, cantidad, img.shape)
    ruido_normal_255 = ruido_normal.astype(np.uint8)
    return cv.add(img, ruido_normal_255)


def girar(img: np.ndarray):
    rows, cols, _ = img.shape
    center = (cols / 2, rows / 2)
    scale = 1
    rotated = []
    for i in range(-60, 60, 7):
        matrix = cv.getRotationMatrix2D(center, i, scale)
        rotated_image = cv.warpAffine(img, matrix, (cols, rows))
        rotated.append(rotated_image)
    return rotated


def main():
    image_dir = os.path.join(BASE_DIR, "imagenes")
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            file = file.split(".")
            filename = file[0]
            extension = file[1]
            if not extension == "png" and not extension == "jpg" and not extension == "jpeg":
                continue
            path = os.path.join(root, filename + "." + extension)
            image = cv.imread(path)
            rotated = girar(image)
            for i, img in enumerate(rotated):
                cv.imwrite(os.path.join(root, filename + "-" + str(i) + "." + extension), img)
                cv.imwrite(os.path.join(root, filename + "-" + str(i) + "ruido" + "." + extension), ruido(img, 6))


if __name__ == '__main__':
    main()
