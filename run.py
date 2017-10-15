from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import data_loader
from lstm import LSTM

# defining image size
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # PyTorch says images must be normalized like this
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

training_set = data_loader.load_data(images='data/train2014', annotations='data/captions_train2014.json', transform=transform)

cnn_encoder = models.vgg16(pretrained=True)
image, targe = training_set[3]
images = []
images.append(image)
images = torch.stack(images, 0)
print(images.size())
# output is a feature vector of size 1000
print(cnn_encoder(data_loader.image_to_variable(images)))

print('Number of samples: ', len(training_set))
img, target = training_set[3]
print("Image Size: ", img.size())
print(target)
