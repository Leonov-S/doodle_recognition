from tkinter import *
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as ts
from doodle_recognition import *

canva_coords = {
    'x1': 150.0 + 15,
    'y1': 107.0 + 15,
    'x2': 850.0 - 15,
    'y2': 817.0 - 15
}

def round_rectangle(x1, y1, x2, y2, radius=25, **kwargs):
        
    points = [x1+radius, y1,
              x1+radius, y1,
              x2-radius, y1,
              x2-radius, y1,
              x2, y1,
              x2, y1+radius,
              x2, y1+radius,
              x2, y2-radius,
              x2, y2-radius,
              x2, y2,
              x2-radius, y2,
              x2-radius, y2,
              x1+radius, y2,
              x1+radius, y2,
              x1, y2,
              x1, y2-radius,
              x1, y2-radius,
              x1, y1+radius,
              x1, y1+radius,
              x1, y1]

    return cv.create_polygon(points, **kwargs, smooth=True)

def clear(e):
    global a
    if e.char == 'c':
        cv.delete(a)
        cv.create_rectangle(
        160.0,
        117.0,
        860.0,
        817.0,
        fill="#FFFFFF",
        outline="")
        draw.rectangle((0, 0, 840, 840), fill='white')

def save(e):
    global loading
    if (loading == True): return
    loading = True
    pix = np.array(image1)
    resized = cv2.resize(pix, (28, 28), interpolation = cv2.INTER_AREA)
    resized = (255-resized)
    data = Image.fromarray(resized)
    data.save('test.png')
    guess = recognize_img(resized, model)
    mainguess.set(guess[0])
    sideguess.set("\n".join(guess[1:]))
    loading = False

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    
    if (x > canva_coords['x1'] and x < canva_coords["x2"] and y > canva_coords["y1"] and y < canva_coords["y2"]
        and lastx > canva_coords['x1'] and lastx < canva_coords["x2"] and lasty > canva_coords['y1'] and lasty < canva_coords['y2']):
        cv.create_line((lastx, lasty, x, y), width=20, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=36)
    draw.line((lastx - canva_coords["x1"], lasty - canva_coords["y1"], x - canva_coords["x1"], y - canva_coords["y1"]), fill='black', width=40)
    lastx, lasty = x, y

model = ts.keras.models.load_model('./dr.h5')
model.load_weights('./drWeight.h5')

loading = False
lastx, lasty = None, None
root = Tk()
root.geometry('1280x960')
root.configure(bg = "#2E0C4F")

cv = Canvas(
    root,
    bg = "#2E0C4F",
    height = 960,
    width = 1280,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
cv.place(x = 0, y = 0)
image1 = Image.new('L', (700, 700), 'white')

draw = ImageDraw.Draw(image1)
mainguess = StringVar()
sideguess = StringVar()
mainguess.set("Draw!")
sideguess.set("waiting..")

round_rectangle(72.0, 67.0, 1192.0, 907.0, radius=20, fill="#581E92")

a= cv.create_rectangle(
    160.0,
    117.0,
    860.0,
    817.0,
    fill="#FFFFFF",
    outline="")

mg_label = Label(root, anchor="nw", textvariable = mainguess, bg = '#581E92',fg = 'white', font=("Inter", 25 * -1))
sg_label = Label(root, anchor="nw", justify=LEFT, textvariable = sideguess, bg = '#581E92',fg = 'grey', font=("Inter", 25 * -1))
mg_label.place(x = 890, y = 117)
sg_label.place(x = 890, y = 145)

cv.create_text(
    216.0,
    835.0,
    anchor="nw",
    text="Left mouse : Draw | Right mouse : Erase | C: Clear",
    fill="#FFFFFF",
    font=("Inter", 25 * -1)
)

cv.bind('<1>', activate_paint)
cv.bind('<ButtonRelease-1>', save)
root.bind('<KeyPress>', clear)

cv.pack(expand=YES, fill=BOTH)
root.resizable(False, False)
root.mainloop()