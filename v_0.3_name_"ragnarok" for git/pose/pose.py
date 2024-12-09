import numpy as np
import csv

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Инициализация весов и смещения
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Обучение персептрона
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Обновление весов и смещения
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        # Функция активации (шаговая функция)
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def print_equation(self):
        # Форматирование и вывод уравнения
        equation = "y = "
        for i, weight in enumerate(self.weights):
            equation += f"{weight:.2f} * x{i+1} + "
        equation += f"{self.bias:.2f}"
        print(equation)

# Пример использования
if __name__ == "__main__":
    with open('data.csv', mode='r') as file:
        lines = file.readlines()
        row_count = len(lines)



    # Пример данных: логическая операция AND
    x = np.zeros((row_count, 2))
    
    # Открываем файл data.csv
    with open('data.csv', mode='r', newline='') as file:
        reader = csv.reader(file)
        
        # Проходим по каждой строке в файле
        i = 0
        for row in reader:
            print(len(row))
            # Проверяем, что строка содержит достаточное количество элементов
            if len(row) >= 2:
                x[i][0] = float(row[6])  # первое значение
                x[i][1] = float(row[7])  # второе значение
                print(f'x1: {x[i][0]}, x2: {x[i][1]}')
                i += 1

    y = np.zeros(row_count)  # Результаты операции AND
    with open('data.csv', mode='r', newline='') as file:
        reader = csv.reader(file)
        
        # Проходим по каждой строке в файле
        i = 0
        for row in reader:
            print(len(row))
            # Проверяем, что строка содержит достаточное количество элементов
            if len(row) >= 3:
                vrem=row[66]
                if vrem=='open':
                    y[i] = 1
                else:
                    y[i]=0
                print(y[i])
                i += 1
    
    # Создание и обучение персептрона
    p = Perceptron(learning_rate=0.1, n_iter=10000)
    p.fit(x, y)

    # Прогнозирование
    predictions = p.predict(x)
    
    #print("Predictions:", predictions)
    res=0
    for z in range(0, len(predictions)):
        print(f"{y[z]} | {predictions[z]}")
        if y[z]==predictions[z]:
            res+=1
    print(f"{res}\{len(predictions)}")

    # Вывод уравнения
    p.print_equation()

