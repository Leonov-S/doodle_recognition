from tkinter import *
import PIL
from PIL import Image, ImageDraw
import cv2

def save():
    global image_number
    filename = f'image_0.png'
    image1.save(filename)
    img = cv2.imread('./image_0.png', cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    cv2.imwrite("rescaled.png", resized)

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

root = Tk()
lastx, lasty = None, None
cv = Canvas(root, width=280, height=280, bg='white')
image1 = PIL.Image.new('RGB', (280, 280), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack()

root.mainloop()