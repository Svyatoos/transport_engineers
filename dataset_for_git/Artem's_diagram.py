import random
import matplotlib.pyplot as plt

mas = [[i for i in range(1, 8)], ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]]

z=0
while z<7:
    mas[0][z]=int(random.random()*100)
    z=z+1


z=0
while z<7:
    print(mas[0][z], " ", mas[1][z]) 
    z=z+1

# Настройка графика
plt.figure(figsize=(10, 5))
plt.bar(mas[1], mas[0])
plt.xlabel('Эмоции')
plt.ylabel('Значения')
plt.title('Диаграмма эмоций')
plt.grid(True)
plt.show()
