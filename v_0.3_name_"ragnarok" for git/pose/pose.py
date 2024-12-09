import numpy as np

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
    # Пример данных: логическая операция AND
    X = np.array([[0, 2.5],
                  [1, 1],
                  [2, -0.5],
                  [3, -2],
                  [4, -3.5],
                  [5, -5],
                  [-3, 7],
                  [7, -8],
                  [8, -9.5],
                  [9, -11],
                  [0, 2],
                  [1, 0.5],
                  [2, -1],
                  [3, -2.5],
                  [4, -4],
                  [5, -5.5],
                  [6, -7],
                  [7, -8.5],
                  [8, -10],
                  [9, -11.5]])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Результаты операции AND
    
    # Создание и обучение персептрона
    p = Perceptron(learning_rate=0.1, n_iter=100)
    p.fit(X, y)

    # Прогнозирование
    predictions = p.predict(X)
    print("Predictions:", predictions)

    # Вывод уравнения
    p.print_equation()
