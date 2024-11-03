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
            print(f"Ошибка при чтении изображения: {image_path}")
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
                #print(f'Landmark {id}: ({cx}, {cy})')  # Отключено для повышения производительности

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # ФреймРейт

        cv2.imshow('python', image)
        cv2.waitKey(50)  # Ждем 1 мс для обновления окна

        return points

if __name__ == "__main__":
    # Указываем путь к папке с изображениями
    folder_path = 'для тестов'

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

        image_names.sort()
        # Вывод массива с именами изображений
        print("Имена изображений:")
        for name in image_names:
            print(name)

        

        with open('dataset.data', 'w', encoding='utf-8') as file:
                for z in range(num_images):
                    image_path = os.path.join(folder_path, image_names[z])
                    processed_points = process_image(image_path)
                    line = ""
                    vrem=image_names[z].split()
                    im=vrem[4].replace('.jpg', '')
                    for id in range(p):
                        line += f"{processed_points[0][id]}, {processed_points[1][id]}; "
                    if im == "нейтральная":
                        im="нейтральное"
                    line+=f"{vrem[2]}, {im}"
                    file.write(line + '\n')

        """while True:
            # Открываем файл для записи данных
            
            if cv2.waitKey(1) == 27:  # exit on ESC
                break
            
            inp=int(input("введите номер файла: "))
            ass=image_names[inp]
            ass=ass.split()
            print(f"{ass[2]} {ass[4]}")"""
        
        

        """for n in range(0, num_images):
                inp=n
                ass=image_names[inp].split()
                im=ass[4].replace('.jpg', '')
                print(f"{ass[1]} {ass[2]} {im}")   
                #print(ass)"""
        
        """for n in range(0, num_images):
            inp=n
            ass=image_names[inp].split()
            im=ass[4].replace('.jpg', '')
            if im!="счастье" and im!="гнев" and im!="грусть" and im!="удивление" and im!="злость" and im!="нейтральное" and im!="отвращение":
                print(f"{ass[1]} {im}") """
        """delta=[]
        de=0
        delta.append("счастье")
        for n in range(0, num_images):
            inp=n
            ass=image_names[inp].split()
            im=ass[4].replace('.jpg', '')
            trig=False
            for g in range(0, de+1):
                if im==delta[g]:
                    trig=True
                    break
            if trig == False:
                de+=1
                delta.append(im)
        for k in delta:
            print(k)"""
        
        cv2.destroyAllWindows()
