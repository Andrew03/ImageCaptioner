from Tkinter import *
from PIL import ImageTk, Image
import tkMessageBox
from pycocotools.coco import COCO

top = Tk()
top.minsize(width=650, height=650)
coco=COCO('data/captions_train2014.json')

E1 = Entry(top, bd = 5)
E1.pack()
get_image_button = Button(top, text="get image(0 - " + str(len(coco.imgs) - 1) + ")", command = lambda: get_image(E1.get())).pack()
captions = "captions will appear here"
image_captions = Label(top, text=captions)
image_captions.pack()
image = ImageTk.PhotoImage(file="assets/temp.png")
panel = Label(top, image = image)
panel.pack(side = "bottom", fill = "both", expand = "yes")

# Getting an image index is not intuitive! 
# It doesn't seem to be ordered the same way that it is in the generic load
def get_image(image_str):
    set_index = int(image_str)
    set_id = coco.imgs.keys()[set_index]
    image_path = coco.imgs[set_id]['file_name']
    annIds = coco.getAnnIds(imgIds=coco.imgs[set_id]['id'])
    captions = coco.loadAnns(annIds)
    image = Image.open("data/train2014/" + image_path)
    image = image.resize((400, 400), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)

    panel.configure(image = image)
    panel.image = image
    caption_string = ""
    for string in captions:
        caption_string += string['caption'] + "\n"
    image_captions.configure(text=caption_string)

top.mainloop()
