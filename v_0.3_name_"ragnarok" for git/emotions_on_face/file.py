import cv2
import time

def show_image_by_number(image_folder, image_number):
    # Путь к изображению
    image_path = f"{image_folder}/{image_number}.jpg"
    
    try:
        # Читаем изображение
        img = cv2.imread(image_path)
        
        if img is not None:
            # Показываем изображение
            cv2.imshow('Image', img)
            
            # Ждем 1000 мс (1 сек), затем окно автоматически закроется
            cv2.waitKey(50)
            cv2.destroyWindow('Image')
        else:
            print(f"Не удалось открыть изображение {image_path}")
    except Exception as e:
        print(f"Произошла ошибка при открытии изображения: {e}")

# Пример использования функции
folder = "/home/svyat/Загрузки/images"  # Указываем путь к папке с изображениями
number = int(input("Введите номер последнего изображения: "))

for i in range(1, number + 1):
    show_image_by_number(folder, str(i))
    #time.sleep(1)