from tkinter import *
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as ts
from doodle_recognition import *


def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill='white')

def save(e):
    global loading
    if (loading == True): return
    loading = True
    pix = np.array(image1)
    resized = cv2.resize(pix, (28, 28), interpolation = cv2.INTER_AREA)
    resized = (255-resized)
    guess = recognize_img(resized, model)
    text.set(guess)
    loading = False

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), fill='black', width=20, capstyle=ROUND, smooth=TRUE, splinesteps=36)
    draw.line((lastx, lasty, x, y), fill='black', width=50)
    lastx, lasty = x, y

model = ts.keras.models.load_model('./dr.h5')
model.load_weights('./drWeight.h5')
loading = False
root = Tk()
lastx, lasty = None, None
cv = Canvas(root, width=840, height=840, bg='white')
image1 = Image.new('L', (840, 840), 'white')
draw = ImageDraw.Draw(image1)
text = StringVar()
text.set("Draw!")

cv.bind('<1>', activate_paint)
cv.bind('<ButtonRelease-1>', save)
cv.pack(expand=YES, fill=BOTH)

text_label = Label(root, textvariable = text, bg = 'white', fg = 'black', font = ('Helvetica', 40))
btn_clear = Button(text="clear", command=clear)
btn_clear.pack()
text_label.pack()

root.mainloop()