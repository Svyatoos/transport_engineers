import threading
from tkinter import *
import cv2
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

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to RGB
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)

    def release(self):
        if self.vid.isOpened():
            self.vid.release()

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Open video source
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            # Convert the image to PhotoImage and display it on the canvas
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.window.after(self.delay, self.update)

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

def on_click():
    print('Закрываем окно...')
    root.destroy()

# Кнопки
btn = Button(frame1, text='Старт', bg='gray', command=lambda: App(root, "Программа для детектирования лица"))
btn.place(relx=0.865, rely=0.034, relheight=0.05, relwidth=0.13)

btn2 = Button(frame1, text='Стоп', bg='gray', command=bbtg2)
btn2.place(relx=0.865, rely=0.089, relheight=0.05, relwidth=0.13)

btn3 = Button(frame1, text='Добавить фоторобот', bg='gray', command=bbtg3)
btn3.place(relx=0.865, rely=0.144, relheight=0.05, relwidth=0.13)

btn4 = Button(frame1, text='Анализ изображения', bg='gray', command=bbtg4)
btn4.place(relx=0.865, rely=0.199, relheight=0.05, relwidth=0.13)

btn5 = Button(frame1, text='Тревога!', bg='red', command=bbtg5)
btn5.place(relx=0.865, rely=0.254, relheight=0.05, relwidth=0.13)

closebtn = Button(frame1, text='Закрыть', bg='gray', command=on_click)
closebtn.place(relx=0.865, rely=0.934 ,relheight=0.05, relwidth=0.13)

root.protocol("WM_DELETE_WINDOW", on_click)
root.mainloop()
