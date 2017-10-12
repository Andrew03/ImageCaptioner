from __future__ import print_function
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import json
import random
import sys
import math
from operator import itemgetter

# import the data set
cap = datasets.CocoCaptions(root = 'data/train2014', annFile = 'data/captions_train2014.json', transform=transforms.ToTensor())
print('Number of samples: ', len(cap))
img, target = cap[3]
print("Image Size: ", img.size())
print(target)
