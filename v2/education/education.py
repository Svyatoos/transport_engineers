import cv2
import numpy as np
import mediapipe as mp
import time
import os
import glob

# Инициализация MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

p = 33  # количество ключевых точек
points = [[i for i in range(1, p + 1)], [i for i in range(1, p + 1)]]  # 0-это x   1-это y


def process_image(image_path):
    global pTime  # Объявляем pTime как глобальную переменную
    pTime = 0  # Инициализируем pTime

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print("Ошибка при чтении изображения.")
            return

        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Вывод координат ключевых точек в консоль
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for id, landmark in enumerate(landmarks):
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                points[0][id] = cx
                points[1][id] = cy
                print(f'Landmark {id}: ({cx}, {cy})')

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # ФреймРейт

        cv2.imshow('python', image)
        cv2.waitKey(0)  # Ждем нажатия любой клавиши для продолжения


if __name__ == "__main__":
    # Указываем путь к папке с изображениями
    folder_path = 'dataset'

    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Папка '{folder_path}' не найдена!")
    else:
        # Получаем список всех файлов в папке, соответствующих расширениям изображений
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp')
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(folder_path, ext)))

        # Количество изображений
        num_images = len(images)
        print(f"В папке найдено {num_images} изображений.")

        # Массив с именами изображений
        image_names = [os.path.basename(img) for img in images]

        # Вывод массива с именами изображений
        print("Имена изображений:")
        for name in image_names:
            print(name)




    image_path = input("Введите путь к изображению: ")
    process_image(image_path)
    with open('dataset.data', 'a', encoding='utf-8') as file:
        for id in range(0, p):
            line = f"{points[0][id]}, {points[1][id]},"
            file.write(line)
    cv2.destroyWindow("python")
    cv2.waitKey(1)