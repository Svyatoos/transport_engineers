import csv

# Открываем файл data.csv
with open('data.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    
    # Проходим по каждой строке в файле
    for row in reader:
        print(len(row))
        # Проверяем, что строка содержит достаточное количество элементов
        if len(row) >= 2:
            x1 = row[0]  # первое значение
            y1 = row[1]  # второе значение
            print(f'x1: {x1}, y1: {y1}')
