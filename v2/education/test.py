import os
import glob

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