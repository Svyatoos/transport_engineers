import cv2
import numpy as np
import mediapipe as mp
import time

# Инициализация MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Подключение камеры
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Length
cap.set(10, 100)  # Brightness

pTime = 0

p=33 #количество ключевых точек
points=[[i for i in range(1, p+1)], [i for i in range(1, p+1)]] #0-это x   1-это y

while True:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                points[0][id]=cx
                points[1][id]=cy
                print(f'Landmark {id}: ({cx}, {cy})')


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # ФреймРейт

        cv2.imshow('python', image)

    if cv2.waitKey(1) == 27:  # exit on ESC
        break

# Завершение работы программы
cv2.destroyWindow("python")
cap.release()
cv2.waitKey(1)