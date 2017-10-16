from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import sys
import data_loader
from lstm import LSTM

def to_var(x):
    return autograd.Variable(x.cuda())

# defining image size
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # PyTorch says images must be normalized like this
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

image_dir, annotation_dir, build_vocab = data_loader.get_file_information()
# training_set = data_loader.load_data(images='data/train2014', annotations='data/captions_train2014.json', transform=transform)
training_set = data_loader.load_data(images=image_dir, annotations=annotation_dir, transform=transform)

cnn_encoder = models.vgg16(pretrained=True).cuda() if torch.cuda.is_available() else (models.vgg16(pretrained=True))
image, target = training_set[3]
images = []
images.append(image)
images = torch.stack(images, 0)
print(images.size())
# output is a feature vector of size 1000
print(cnn_encoder(data_loader.image_to_variable(images)))
img2, tar2 = training_set[100]
print(img2.size())
#print(cnn_encoder(to_var(images)))

'''
# creating the model
batch_size, min_occurrences = 32, 10
D_embed, H, D_out = 30, 100,30
#model = LSTM(D_embed, H, len(word_to_index), batch_size).cuda() if torch.cuda.is_available() else LSTM(D_embed, H, len(word_to_index), batch_size)
model = LSTM(D_embed, H, 1000, batch_size).cuda() if torch.cuda.is_available() else LSTM(D_embed, H, 1000, batch_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
'''

print('Number of samples: ', len(training_set))
img, target = training_set[3]
print("Image Size: ", img.size())
print(target)

word_to_index, index_to_word  = data_loader.create_vocab(training_set, min_occurrence=10)
data_loader.write_vocab_to_file(index_to_word)
