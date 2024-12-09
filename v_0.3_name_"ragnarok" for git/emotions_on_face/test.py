import cv2
import os

# Укажите путь к папке с фотографиями и к текстовому файлу
folder_path = '/home/svyat/Загрузки/images'  # Замените на путь к вашей папке с изображениями
txt_file_path = 'number.txt'  # Замените на путь к вашему .txt файлу

# Читаем имена файлов из .txt файла
with open(txt_file_path, 'r') as file:
    image_names = [line.strip() for line in file.readlines()]

# Проходим по каждому имени файла и отображаем изображение
for image_name in image_names:
    image_path = os.path.join(folder_path, f"{image_name}.jpg")
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        continue

    # Показываем изображение
    cv2.imshow('Image', image)
    
    # Ждем нажатия клавиши для перехода к следующему изображению
    key = cv2.waitKey(0)  # 0 - ждать бесконечно, пока не будет нажата клавиша
    if key == 27:  # ESC для выхода
        break

# Закрываем все окна после завершения показа изображений
cv2.destroyAllWindows()
