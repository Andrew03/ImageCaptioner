from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as autograd
import torchvision.models as models
import sys
import data_loader
from lstm import LSTM
from encoder import EncoderCNN

# defining image size
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # PyTorch says images must be normalized like this
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

image_dir = ""
annotation_dir = ""
build_vocab = False
if len(sys.argv) == 1:
    image_dir, annotation_dir, build_vocab = data_loader.get_file_information()
elif len(sys.argv) < 3:
    print("Include image data set and caption data set")
    sys.exit()
else:
    image_dir = sys.argv[1]
    annotation_dir = sys.argv[2]
    if len(sys.argv) > 3:
        build_vocab = True
    

# training_set = data_loader.load_data(images='data/train2014', annotations='data/captions_train2014.json', transform=transform)
training_set = data_loader.load_data(images=image_dir, annotations=annotation_dir, transform=transform)

# CNN is vgg16 with batch normalization
# Doesn't seem like vgg16 with batch normalization works right now... might be me needing to update pytorch
# cnn_encoder = models.vgg16_bn(pretrained=True).cuda() if torch.cuda.is_available() else (models.vgg16_bn(pretrained=True))
#cnn_encoder = models.vgg16(pretrained=True).cuda() if torch.cuda.is_available() else (models.vgg16(pretrained=True))
# think this is an easier way of using cuda when available but should check

# rebuilds vocabulary if necessary or specified
# otherwise, uses the already prebuilt vocabulary
print("rebuilding vocabulary" if build_vocab == True else "using old vocabulary", file=sys.stderr)
word_to_index, index_to_word  = data_loader.create_vocab(training_set, min_occurrence=5) if build_vocab == True else (data_loader.load_vocab())
# overwrites the prebuilt vocabulary if specified, otherwise stores the vocabulary
if build_vocab == True:
    data_loader.write_vocab_to_file(index_to_word)
