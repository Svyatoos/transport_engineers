# подключаем библиотеки
import cv2
#import matplotlib.pyplot as plt
#from deepface import DeepFace


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
            #cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)

    # возвращаем кадр с рамками
    
    return frameOpencvDnn,faceBoxes

# загружаем веса для распознавания лиц
faceProto="opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel="opencv_face_detector_uint8.pb"

# запускаем нейросеть по распознаванию лиц
faceNet=cv2.dnn.readNet(faceModel,faceProto)

# получаем видео с камеры
video=cv2.VideoCapture(0)
# пока не нажата любая клавиша — выполняем цикл
"""
while cv2.waitKey(1)<0:
    # получаем очередной кадр с камеры
    hasFrame,frame=video.read()
    # если кадра нет
    if not hasFrame:
        # останавливаемся и выходим из цикла
        cv2.waitKey()
        break
    # распознаём лица в кадре
    resultImg,faceBoxes=highlightFace(faceNet,frame)

    # если лиц нет
    if not faceBoxes:
        # выводим в консоли, что лицо не найдено
        print("Лица не распознаны")

    i=0
    imageFace = resultImg[faceBoxes[i][1]-10 : faceBoxes[i][3]+10, faceBoxes[i][0]-10 : faceBoxes[i][2]+10]
    
    #    imageFace это переменная с изоброжением 
    #    там горантировоно тока лицо
    #    тута твой код
"""
def face_id(img_1, folder_img_2):
    try:
        result = DeepFace.find(img_path = img_1, db_path = folder_img_2)
        return result
    except Exception as ex:
        return ex
if __name__ == '__main__':
    img1_path = input("Введите путь к первому изображению: ")
    folder_img2_path = input("Введите путь к папке с изображениями: ")
    face_similar = face_id(img1_path, folder_img2_path)
    print(face_similar)
    
   #1 cv2.imshow("Face detection", imageFace) #это вывод изоброжения литца    
    
    
    
    
    
    
    
    
    
    
    #hello
    #
    #
    
   
