from Tkinter import *
import tkMessageBox
import data_loader
import torchvision.transforms as transforms

def generate_button_callback():
    tkMessageBox.showinfo( "Hello Python", "Hello World")

def get_image(image_str):
    tkMessageBox.showinfo("", image_str)



transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

top = Tk()
training_set = data_loader.load_data(images='data/training2014', annotations='data/captions_train2014.json', transform=transform)
image_data = data_loader.load_image_information('data/instances_train2014.json')
print("finished loading json")
# displays the window
generate_button = Button(top, text ="Generate a caption!", command = generate_button_callback)
generate_button.pack()
L1 = Label(top, text="Image Name")
L1.pack(side = LEFT)
E1 = Entry(top, bd = 5)
get_image_button = Button(top, text="get image", command = lambda: get_image(E1.get()))
get_image_button.pack()

E1.pack(side = RIGHT)
top.mainloop()
