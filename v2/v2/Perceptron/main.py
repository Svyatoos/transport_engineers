import numpy as np
from typing import List

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Обучение персептрона."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Обучение модели
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Обновление весов
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        """Активационная функция (пороговая функция)."""
        return np.where(x >= 0, 1, 0)

    def predict(self, X: np.ndarray):
        """Предсказываем класс."""
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

# Пример использования персептрона
if __name__ == "__main__":
    # Тренировочные данные (логическая операция И)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])  # Ожидаемые выходы

    # Создание и обучение персептрона
    p = Perceptron(learning_rate=0.1, n_iter=10)
    p.fit(X, y)

    # Предсказание
    print("Предсказания:")
    print(p.predict(X))
