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
batch_size = 30
images = []
for i in range(batch_size):
    image, target = training_set[3]
    images.append(image)
images = torch.stack(images, 0)
print(images.size())

print('Number of samples: ', len(training_set))
img, target = training_set[3]
print("Image Size: ", img.size())
print(target)

# rebuilds vocabulary if necessary or specified
# otherwise, uses the already prebuilt vocabulary
print("rebuilding vocabulary" if build_vocab == True else "using old vocabulary")
word_to_index, index_to_word  = data_loader.create_vocab(training_set, min_occurrence=5) if build_vocab == True else (data_loader.load_vocab())
# overwrites the prebuilt vocabulary if specified, otherwise stores the vocabulary
if build_vocab == True:
    data_loader.write_vocab_to_file(index_to_word)

# CNN takes in image and passes feature vector to RNN once at time step -1
# Then RNN takes in word at time step i and tries to make that word more probable 
# What does it mean for the image and words to be mapped to the same space?
# Does that mean we combine the image feature vector and the word vector?

# creating the model
batch_size, min_occurrences = 32, 10
D_embed, H, D_out = 32, 100,30

encoder_cnn = EncoderCNN(D_embed)
model = LSTM(D_embed, H, len(word_to_index), batch_size).cuda() if torch.cuda.is_available() else LSTM(D_embed, H, len(word_to_index), batch_size)
if torch.cuda.is_available():
    model.cuda()
loss_function = nn.NLLLoss()
# try using adams
#optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    # resetting gradients and hidden layer values
    model.zero_grad()
    model.hidden = model.init_hidden()

    images, captions = data_loader.create_batch(training_set, word_to_index, 5, batch_size=batch_size)
    input_captions = data_loader.create_input_batch_captions(captions)
    target_captions = data_loader.create_target_batch_captions(captions)
    
    image_features = encoder_cnn(images)
    input_images = data_loader.create_input_batch_image_features(image_features, D_embed)
    
    caption_scores = model(input_captions)
    loss = loss_function(caption_scores, target_captions)

    print(str(epoch) + ", score: " + str(loss.data.select(0, 0) / batch_size), file=sys.stderr)
    loss.backward()
    optimizer.step()
