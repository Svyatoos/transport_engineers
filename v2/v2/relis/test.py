import tkinter as tk
from tkinter import messagebox

def show_popup():
    # Создаем всплывающее окно
    messagebox.showinfo("Заголовок", "Это ваше всплывающее сообщение!")

# Создаем главное окно
root = tk.Tk()
root.title("Главное окно")
root.geometry("300x200")  # Размер окна

im="del.jpg"

label = tk.Label(root, text=f"Обнаружен приступник\n {im }")
label.pack()

# Запускаем главный цикл приложения
root.mainloop()
