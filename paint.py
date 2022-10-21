from tkinter import *
import PIL
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as ts
import time

from doodle_recognition import *

def save(model):
    global image_number
    filename = f'image_0.png'
    image1.save(filename)
    img = cv2.imread('./image_0.png', cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    resized = (255-resized)
    cv2.imwrite("rescaled.png", resized)
    time.sleep(1)
    im = Image.open('rescaled.png')
    numpy_data = np.asarray(im.convert('1'))
    recognize_img(numpy_data, model)


def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_oval(lastx, lasty, x, y, fill='black', width=10)
    draw.line((lastx, lasty, x, y), fill='black', width=12)
    lastx, lasty = x, y

model = ts.keras.models.load_model('./dr.h5')
model.load_weights('./drWeight.h5')
root = Tk()
lastx, lasty = None, None
cv = Canvas(root, width=280, height=280, bg='white')
image1 = PIL.Image.new('RGB', (280, 280), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command= lambda: save(model))
btn_save.pack()

root.mainloop()