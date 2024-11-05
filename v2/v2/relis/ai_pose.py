import cv2
import numpy as np
import mediapipe as mp
import time
import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
                #print(f'Landmark {id}: ({cx}, {cy})')  

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # ФреймРейт

        cv2.imshow('python', image)
        cv2.waitKey(0)  # Ждем 1 мс для обновления окна

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

if __name__ == "__main__":
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


    


    image=input("введите название изображения: ")
    vrem=process_image(image)
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


    cv2.destroyAllWindows()

    