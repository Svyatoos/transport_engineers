import os

# Указываем путь к папке с фотографиями
folder_path = r'images'

# Получаем список всех файлов в папке
files = os.listdir(folder_path)

# Фильтруем только файлы изображений (расширения jpg, jpeg, png)
images = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Сортируем файлы для предсказуемого порядка переименования
images.sort()

# Перебираем файлы и переименовываем их
for i, image in enumerate(images):
    old_name = os.path.join(folder_path, image)
    new_name = f"{i + 1}.jpg"  # Можно заменить расширение на любое другое, если нужно
    new_full_path = os.path.join(folder_path, new_name)
    
    # Переименовываем файл
    os.rename(old_name, new_full_path)