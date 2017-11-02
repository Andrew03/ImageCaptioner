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
'''
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
'''
    

#training_set = data_loader.load_data(images=image_dir, annotations=annotation_dir, transform=transform)
train_set = data_loader.load_data(images='data/train2014', annotations='data/annotations/captions_train2014.json', transform=transform)
val_set = data_loader.load_data(images='data/val2014', annotations='data/annotations/captions_val2014.json', transform=transform)
# rebuilds vocabulary if necessary or specified
# otherwise, uses the already prebuilt vocabulary
print("rebuilding vocabulary" if build_vocab == True else "using old vocabulary", file=sys.stderr)
word_to_index, index_to_word  = data_loader.create_vocab(train, min_occurrence=5) if build_vocab == True else (data_loader.load_vocab())
# overwrites the prebuilt vocabulary if specified, otherwise stores the vocabulary
if build_vocab == True:
  data_loader.write_vocab_to_file(index_to_word)
# batch the data
#batched_train_set = data_loader.batch_data(train_set, word_to_index, batch_size=32)
#batched_val_set = data_loader.batch_data(val_set, word_to_index, batch_size=32)
batched_train_set = data_loader.load_batched_data('batched_train_set.txt', word_to_index)
batched_val_set = data_loader.load_batched_data('batched_val_set.txt', word_to_index)
if batched_train_set == None:
  batched_train_set = data_loader.batch_data(train_set, word_to_index, batch_size=32)
  data_loader.write_batched_data(batched_train_set, file_name="batched_train_set.txt")
if batched_val_set == None:
  batched_val_set = data_loader.batch_data(val_set, word_to_index, batch_size=32)
  data_loader.write_batched_data(batched_val_set, file_name="batched_val_set.txt")

# CNN is vgg16 with batch normalization
# Doesn't seem like vgg16 with batch normalization works right now... might be me needing to update pytorch
# cnn_encoder = models.vgg16_bn(pretrained=True).cuda() if torch.cuda.is_available() else (models.vgg16_bn(pretrained=True))
#cnn_encoder = models.vgg16(pretrained=True).cuda() if torch.cuda.is_available() else (models.vgg16(pretrained=True))
# think this is an easier way of using cuda when available but should check

#print('Number of samples: ', len(training_set))
#img, target = training_set[3]
#print("Image Size: ", img.size())
#print(target)


# CNN takes in image and passes feature vector to RNN once at time step -1
# Then RNN takes in word at time step i and tries to make that word more probable 
# What does it mean for the image and words to be mapped to the same space?
# Does that mean we combine the image feature vector and the word vector?

# creating the model
batch_size, min_occurrences = 32, 10
D_embed, H, D_out = 32, 124,32

encoder_cnn = EncoderCNN(D_embed)
model = LSTM(D_embed, H, len(word_to_index), batch_size)
if torch.cuda.is_available():
  model.cuda()
loss_function = nn.NLLLoss()
# tried using 0.001
optimizer = optim.Adam(model.parameters(), lr=0.0005)

record_error = False
if len(sys.argv) > 1:
  record_error = True

error_file = open(sys.argv[1], 'w') if record_error == True else None

for epoch in range(5000):
  # resetting gradients and hidden layer values
  model.zero_grad()
  model.hidden = model.init_hidden()

  # training set
  train_images, train_captions = data_loader.create_batch(train_set, batched_train_set, word_to_index, 5, batch_size=batch_size)
  train_input_captions = data_loader.create_input_batch_captions(train_captions)
  train_target_captions = data_loader.create_target_batch_captions(train_captions)
  # validation set
  val_images, val_captions = data_loader.create_batch(val_set, batched_val_set, word_to_index, 5, batch_size=batch_size)
  val_input_captions = data_loader.create_input_batch_captions(val_captions)
  val_target_captions = data_loader.create_target_batch_captions(val_captions)

  # training
  train_image_features = encoder_cnn(train_images)
  train_image_features = autograd.Variable(train_image_features.data)
  # do i need to store initial score?
  initial_score = model(train_image_features)
  
  train_caption_scores = model(train_input_captions)
  loss = loss_function(train_caption_scores, train_target_captions)

  if record_error == True:
    error_file.write(str(loss.data.select(0, 0) / batch_size) + "\n")
  #print(str(loss.data.select(0, 0) / batch_size))
  print(str(epoch) + ", score: " + str(loss.data.select(0, 0) / batch_size), file=sys.stderr)
  loss.backward()
  optimizer.step()

  model.hidden = model.init_hidden()
  # validating
  val_image_features = encoder_cnn(val_images)
  # do i need to store initial score?
  initial_score = model(val_image_features)
  
  val_caption_scores = model(val_input_captions)
  loss = loss_function(val_caption_scores, val_target_captions)
  if record_error == True:
    error_file.write(str(loss.data.select(0, 0) / batch_size) + "\n")
  #print(str(loss.data.select(0, 0) / batch_size))
  print(str(epoch) + ", score: " + str(loss.data.select(0, 0) / batch_size), file=sys.stderr)
error_file.close()
torch.save(model.state_dict(), 'model/model5.01.pt')
