import threading
from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk

root = Tk()
root['bg'] = '#fff'
root.title('Программа для детектирования лица')
root.geometry('1500x800')
root.resizable(width=False, height=False)

# Фреймы

frame1 = Frame(root, bg='white')
frame1.place(relwidth=1, relheight=1)

frame2 = Frame(root, bg='gray')
frame2.place(relx=0.01, rely=0.034, relwidth=0.85, relheight=0.95)

# Изображение

cs = Canvas(root, height=400, width=700)
image = Image.open("3.jpg")  # Убедитесь, что файл 3.jpg находится в той же директории, где выполняется скрипт
photo = ImageTk.PhotoImage(image)
image = cs.create_image(0, 0, anchor='nw', image=photo)
cs.place(relx=0.01, rely=0.034, relwidth=0.85, relheight=0.95)

# Глобальная переменная для управления потоком
running = False

def start_loop():
    global running
    while running:
        print(1)
        root.update_idletasks()  # Обновление интерфейса
        root.after(50)  # Задержка перед следующей итерацией

def bbtg():
    global running
    if not running:
        running = True
        threading.Thread(target=start_loop).start()

def bbtg2():
    global running
    running = False
    print("stop")

def bbtg3():
    print('Добавляем фоторобот...')

def bbtg4():
    print('Производим анализ изображения...')

def bbtg5():
    print('Тревога!')

# Кнопки

btn = Button(frame1, text='Старт', bg='gray', command=bbtg)
btn.place(relx=0.865, rely=0.034, relheight=0.05, relwidth=0.13)

btn2 = Button(frame1, text='Стоп', bg='gray', command=bbtg2)
btn2.place(relx=0.865, rely=0.089, relheight=0.05, relwidth=0.13)

btn3 = Button(frame1, text='Добавить фоторобот', bg='gray', command=bbtg3)
btn3.place(relx=0.865, rely=0.144, relheight=0.05, relwidth=0.13)

btn4 = Button(frame1, text='Анализ изображения', bg='gray', command=bbtg4)
btn4.place(relx=0.865, rely=0.199, relheight=0.05, relwidth=0.13)

btn5 = Button(frame1, text='Тревога!', bg='red', command=bbtg5)
btn5.place(relx=0.865, rely=0.254, relheight=0.05, relwidth=0.13)

root.mainloop()