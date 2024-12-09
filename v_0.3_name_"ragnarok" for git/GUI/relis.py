# подключаем библиотеки
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import time
import numpy as np
import mediapipe as mp
import time
import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox


# Инициализация MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

p = 33  # количество ключевых точек
points = [[i for i in range(1, p + 1)], [i for i in range(1, p + 1)]]  # 0-это x   1-это y




"""функции нейронки"""



def process_image(image_path):
    global pTime  # Объявляем pTime как глобальную переменную
    pTime = 0  # Инициализируем pTime

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Загрузка изображения
        #image = cv2.imread(image_path)
        image = image_path
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
                #print(f'Landmark {id}: ({cx}, {cy})')  

        """cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # ФреймРейт

        cv2.imshow('python', image)
        cv2.waitKey(0)  # Ждем 1 мс для обновления окна
"""
        return points

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    rows = []
    for line in lines:
        row = line.strip().split(',')
        
        if len(row) == 67:  # Проверяем, чтобы было ровно 67 значений
            rows.append([float(x) if x not in ['открытая', 'закрытая'] else x for x in row])
            
    return rows



"""конец"""

# функция определения лиц
def highlightFace(net, frame, conf_threshold=0.7):
    # делаем копию текущего кадра
    frameOpencvDnn=frame.copy()
    # высота и ширина кадра
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # преобразуем картинку в двоичный пиксельный объект
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    # устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # выполняем прямой проход для распознавания лиц
    detections=net.forward()
    # переменная для рамок вокруг лица
    faceBoxes=[]
    # перебираем все блоки после распознавания
    for i in range(detections.shape[2]):
        # получаем результат вычислений для очередного элемента
        confidence=detections[0,0,i,2]
        # если результат превышает порог срабатывания — это лицо
        if confidence>conf_threshold:
            # формируем координаты рамки
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            # добавляем их в общую переменную
            faceBoxes.append([x1,y1,x2,y2])
            
            
            # рисуем рамку на кадре
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)

    # возвращаем кадр с рамками
    
    return frameOpencvDnn,faceBoxes








"""

# Путь к папке с фотографиями
photos_folder = "criminal"

# Получаем список всех изображений в папке
photos = [os.path.join(photos_folder, f) for f in os.listdir(photos_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

def face_id(img_1, img_2):
    try:
        result = DeepFace.verify(img_1, img_2)
        return result['verified']
    except Exception as e:
        print(f"Error during face verification: {e}")
        return False"""










rows = read_data('data.csv')
print(rows)
if not rows:  # Проверка, что данные были успешно прочитаны
    raise ValueError("Файл не содержит данных.")

# Создание DataFrame
df = pd.DataFrame(rows)
print(df)

# Назначение имен столбцов
column_names = ['x' + str(i+1) for i in range(33)] + ['y' + str(i+1) for i in range(33)] + ['target']
print(column_names)
df.columns = column_names

# Извлечение признаков и целевой переменной
X = df.iloc[:, :-1]
y = df['target']

# Преобразование целевой переменной в числовой формат
y = y.apply(lambda x: 1 if x == 'открытая' else 0)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на тренировочные и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Линейная регрессия
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка модели
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Точность на тренировочном наборе: {train_score:.4f}")
print(f"Точность на тестовом наборе: {test_score:.4f}")





















# загружаем веса для распознавания лиц
faceProto="opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel="opencv_face_detector_uint8.pb"

# запускаем нейросеть по распознаванию лиц
faceNet=cv2.dnn.readNet(faceModel,faceProto)

# получаем видео с камеры
video=cv2.VideoCapture(0)






frame_count = 0

# пока не нажата любая клавиша — выполняем цикл
while cv2.waitKey(1)<0:
    frame_count+=1

    # получаем очередной кадр с камеры
    hasFrame,frame=video.read()
    # если кадра нет
    if not hasFrame:
        # останавливаемся и выходим из цикла
        cv2.waitKey()
        break


    """поза"""

    vrem=process_image(frame)
    res=[0]*66
    j=0
    w=0
    
    while j<66:
        
        res[j]=vrem[0][w]
        res[j+1]=vrem[1][w]
        j+=2
        w+=1
    print(res)

    # Преобразуем в нужный формат (2D массив)
    new_data_reshaped = np.array(res).reshape(1, -1)

    # Нормализация новых данных
    new_data_scaled = scaler.transform(new_data_reshaped)

    # Предсказание
    prediction = model.predict(new_data_scaled)
    predicted_class = 'открытая' if prediction[0] >= 0.5 else 'закрытая'
    print(f"Предсказанный класс: {predicted_class}")

    """конец поз"""


    """сравнение лиц"""
    """if frame_count % 100==0:
        frame_count=0
        for photo_path in photos:
            if face_id(photo_path, "current_frame.jpg"):
                print(f"Match found with: {photo_path}")
                root = tk.Tk()
                root.title("Главное окно")
                root.geometry("300x200")  # Размер окна

                

                label = tk.Label(root, text=f"Обнаружен приступник\n {photo_path }")
                label.pack()

                # Запускаем главный цикл приложения
                root.mainloop()"""

    """конец сравнения лиц"""

    # распознаём лица в кадре
    resultImg,faceBoxes=highlightFace(faceNet,frame)

    # если лиц нет
    if not faceBoxes:
        # выводим в консоли, что лицо не найдено
        print("Лица не распознаны")

    #если есть лицо проверяем эмоцию
    
    else:
        len_faceBox=len(faceBoxes)
        i=0
        frameHeight, frameWidth, _ = frame.shape
        while i<len_faceBox:
            if (faceBoxes[i][1]-10 >= 0) and (faceBoxes[i][0]-10 >= 0) and (faceBoxes[i][3]+10 <= frameHeight) and (faceBoxes[i][3]+10 <= frameWidth):
                
                
                imageFace = resultImg[faceBoxes[i][1]-10 : faceBoxes[i][3]+10, faceBoxes[i][0]-10 : faceBoxes[i][2]+10]
                
                if imageFace is None:
                    raise Exception("Ошибка: изображение не загружено. Проверьте путь к файлу.")
                #cv2.imshow("Face detection", imageFace)

            
                
                # Анализ изображения с помощью DeepFace
                analyze = DeepFace.analyze(imageFace, actions = ['emotion'], enforce_detection=False)
                print(analyze)
                
                angry=0.1
                disgust=0.1
                fear=0.1
                happy=0.1
                sad=0.1
                surprise=0.1
                neutral=0.1
                #узнаём какая эмоция преобладает
                for person in analyze:
                    if 'emotion' in person and 'angry' in person['emotion']:
                        angry_confidence = person['emotion']['angry']
                        angry=angry_confidence
                        
 
                    if 'emotion' in person and 'disgust' in person['emotion']:
                        angry_confidence = person['emotion']['disgust']
                        disgust=angry_confidence    
                    if 'emotion' in person and 'fear' in person['emotion']:
                        angry_confidence = person['emotion']['fear']
                        fear=angry_confidence
                    
                    if 'emotion' in person and 'happy' in person['emotion']:
                        angry_confidence = person['emotion']['happy']
                        happy=angry_confidence
                    
                    if 'emotion' in person and 'sad' in person['emotion']:
                        angry_confidence = person['emotion']['sad']
                        sad=angry_confidence
                    
                    if 'emotion' in person and 'surprise' in person['emotion']:
                        angry_confidence = person['emotion']['surprise']
                        surprise=angry_confidence
                    
                    if 'emotion' in person and 'neutral' in person['emotion']:
                        angry_confidence = person['emotion']['neutral']
                        neutral=angry_confidence
                
                
                variables = {
                    'angry': angry,
                    'disgust': disgust,
                    'fear': fear,
                    'happy': happy,
                    'sad': sad,
                    'surprise': surprise,
                    'neutral': neutral
                }

                # Находим переменную с максимальным значением
                #max_var = max(variables, key=variables.get)
                #max_value = variables[max_var]
                
                second_largest_value2 = sorted(set(variables.values()), reverse=True)[1]
                second_largest_value1 = sorted(set(variables.values()), reverse=True)[0]
        # Находим имя ключа с этим значением
                two_name = next(key for key, value in variables.items() if value == second_largest_value2)
                one_name = next(key for key, value in variables.items() if value == second_largest_value1)

                

                #сортировка списка
                sorted(variables)

                # Выводим результат
                #print(f"Переменная с максимальным значением: {max_var} (значение: {max_value})")


                #cv2.putText(resultImg, str(max_var), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 2)
                #cv2.putText(resultImg, f"{one_name}-{two_name}", (faceBoxes[i][0],faceBoxes[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 2)
                if predicted_class=="закрытая":
                    predicted_class_eng="closed"
                elif predicted_class=="открытая":
                    predicted_class_eng="open"
                cv2.putText(resultImg, f"{one_name} {predicted_class_eng}", (faceBoxes[i][0],faceBoxes[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 2)

                
                
            i+=1


    
    
    # выводим картинку с каме   ры
    
    cv2.imshow("Face detection", resultImg)
    #cv2.imshow("Face detection", )
    
    #time.sleep(1)




