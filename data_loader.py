import torch
import torch.autograd as autograd
import torchvision.datasets as datasets

def load_data(images, annotations, transform):
    return datasets.CocoCaptions(root = images, annFile = annotations, transform = transform)

def image_to_variable(image):
    if torch.cuda.is_available():
        image = image.cuda()
    return autograd.Variable(image)
