import os
import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# Определение класса Perceptron как в вашем коде...

class Perceptron:       #класс персептрон
    eta: float          #скорость обучения между 0.0 и 1.0
    n_iter: int         # количество проходов для обучения
    random_state: int   #Опорное значение генератора случайных чисел для инициализации весов

    #Атрибуты

   # w_: ld-array        #Веса после подгонки
    #b_: Scalar          #Смещение после подгонки
    errors_: list       #Количество неправильных классификаций (обновлений) в каждой эпохе

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self, x, y):
        """

        Соответствие тренировочным данным.

        Параметры:

            x: array-like, shape = [n_examples, n_features] 
            обучающий вектор, где n_examples это количество образцов
                                а n_features это количество признаков

            y: array-like, shape = [n_examples]
            целевые значения

        возвращаемые значения:
            self: object

        """
        rgen=np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self.b_ = np.float_(0.)
        self.errors_=[]

        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(x, y):
                update=self.eta*(target-self.predict(xi))
                self.w_ += update*xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, x):
        #вычесление актического вывода
        return np.dot(x, self.w_)+self.b_
    def predict(self, x):
        #Возвращает метки класса после шага
        return np.where(self.net_input(x)>=0.0, 1, 0)





s = 'iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
x = df.iloc[0:100, [0, 2]].values

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='s', label='Versicolor')

plt.xlabel('Length of sepal [cm]')
plt.ylabel('Length of petal [cm]')
plt.legend(loc='upper left')
plt.show()

ppn=Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Количество обновлений')
plt.show()

print("OK")

