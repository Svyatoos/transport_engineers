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



folder = "/home/svyat/Загрузки/images"  # Указываем путь к папке с изображениями
number = int(input("Введите номер последнего изображения: "))
i=1



# получаем видео с камеры

# пока не нажата любая клавиша — выполняем цикл
with open('number.txt', 'r') as file:
    for line in file:

        num=line.strip()
        image_path = f"{folder}/{num}.jpg"

        frame=cv2.imread(image_path)
        # получаем очередной кадр с камеры
        
       
        # если кадра нет
        
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
                    
                    print("1")
                    imageFace = resultImg[faceBoxes[i][1]-10 : faceBoxes[i][3]+10, faceBoxes[i][0]-10 : faceBoxes[i][2]+10]
                    
                    if imageFace is None:
                        raise Exception("Ошибка: изображение не загружено. Проверьте путь к файлу.")
                    print("2")
                    
                    print(len_faceBox)
                    print(faceBoxes)
                    #cv2.imshow("Face detection", imageFace)

                
                    
                    # Анализ изображения с помощью DeepFace
                    analyze = DeepFace.analyze(imageFace, actions = ['emotion'], enforce_detection=False)
                    print(analyze)
                    print("3")
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
                    print("4")
                    print(angry)
                    print(disgust)
                    print(fear)
                    print(happy)
                    print(sad)
                    print(surprise)
                    print(neutral)
                    print("5")
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
                    print("6")
                    second_largest_value2 = sorted(set(variables.values()), reverse=True)[1]
                    second_largest_value1 = sorted(set(variables.values()), reverse=True)[0]
            # Находим имя ключа с этим значением
                    two_name = next(key for key, value in variables.items() if value == second_largest_value2)
                    one_name = next(key for key, value in variables.items() if value == second_largest_value1)

                    print("7")

                    #сортировка списка
                    sorted(variables)

                    # Выводим результат
                    #print(f"Переменная с максимальным значением: {max_var} (значение: {max_value})")


                    #cv2.putText(resultImg, str(max_var), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 2)
                    #cv2.putText(resultImg, f"{one_name}-{two_name}", (faceBoxes[i][0],faceBoxes[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 2)
                    cv2.putText(resultImg, f"{one_name}", (faceBoxes[i][0],faceBoxes[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 2)
    
                    print("8")
                    
                i+=1


        # выводим картинку с камеры
        
        print(num)

        cv2.imshow(num, resultImg)
        cv2.waitKey(0)
        cv2.destroyWindow(num)
        i+=1
        if i>=number:
            break

