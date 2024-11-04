# подключаем библиотеки
import cv2
#import matplotlib.pyplot as plt
from deepface import DeepFace

def face_id(img_1, img_2):
    try:
        result = DeepFace.verify(img1_path = img_1, img2_path = img_2)
        return result
    except Exception as ex:
        return ex

if __name__ == '__main__':
    img1_path = input("Введите путь к первому изображению: ")
    img2_path = input("Введите путь к второму изображению: ")
    face_verify = face_id(img1_path, img2_path)
    print(face_verify)
