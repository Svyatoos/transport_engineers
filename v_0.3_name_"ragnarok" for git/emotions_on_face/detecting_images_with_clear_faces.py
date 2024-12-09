# подключаем библиотеки
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import time

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

# загружаем веса для распознавания лиц
faceProto="opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel="opencv_face_detector_uint8.pb"

# запускаем нейросеть по распознаванию лиц
faceNet=cv2.dnn.readNet(faceModel,faceProto)

# пока не нажата любая клавиша — выполняем цикл


f = open('number.txt', 'w')

folder = "/home/svyat/Загрузки/images"  # Указываем путь к папке с изображениями
number = int(input("Введите номер последнего изображения: "))

for i in range(1, number+1):
    # получаем очередной кадр с камеры
    image_path = f"{folder}/{i}.jpg"
    frame=cv2.imread(image_path)

    # распознаём лица в кадре
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    # если лиц нет
    if faceBoxes:
        # выводим в консоли, что лицо не найдено
      
        f.write(str(i)+'\n')
    print(str(i))
        
    # выводим картинку с камеры
    #cv2.imshow(str(i), resultImg)
    #cv2.waitKey(10)
    #cv2.destroyWindow(str(i))
