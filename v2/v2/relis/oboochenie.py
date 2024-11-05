import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    rows = []
    for line in lines:
        row = line.strip().split(',')
        
        if len(row) == 67:  # Проверяем, чтобы было ровно 67 значений
            rows.append([float(x) if x not in ['открытая', 'закрытая'] else x for x in row])
            
    return rows

# Чтение данных из файла
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

# Предположим, у вас есть новые данные (без строковых значений)




"""new_data = [393,504,406,494,408,496,412,498,403,492,403,492,404,491,431,519,426,510,394,526,391,522,426,612,421,585,299,661,329,604,285,617,292,605,281,629,289,626,301,622,311,624,297,625,316,620,410,801,410,766,317,634,320,617,247,789,255,754,258,821,263,785,167,812,190,776]
  

# Преобразуем в нужный формат (2D массив)
new_data_reshaped = np.array(new_data).reshape(1, -1)

# Нормализация новых данных
new_data_scaled = scaler.transform(new_data_reshaped)

# Предсказание
prediction = model.predict(new_data_scaled)
predicted_class = 'открытая' if prediction[0] >= 0.5 else 'закрытая'
print(f"Предсказанный класс: {predicted_class}")"""

error=0
inp=read_data('data.csv')
for strok in range(0, 150):
    a=inp[strok]
    sravn=a[66]
    a.pop()
    # Преобразуем в нужный формат (2D массив)
    new_data_reshaped = np.array(a).reshape(1, -1)

    # Нормализация новых данных
    new_data_scaled = scaler.transform(new_data_reshaped)
 
    # Предсказание
    prediction = model.predict(new_data_scaled)
    predicted_class = 'открытая' if prediction[0] >= 0.5 else 'закрытая'
    #print(f"Предсказанный класс: {predicted_class}")
    if predicted_class != sravn:
        error+=1
        #print("номер фото где найдена ошибка:")
    print(f"количество ошибок : {error}")

 

    

