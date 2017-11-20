from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as autograd
import torchvision.models as models
import sys
import random
import data_loader
import trainer
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

build_vocab = False

train_set = data_loader.load_data(images='data/train2014', annotations='data/annotations/captions_train2014.json', transform=transform)
val_set = data_loader.load_data(images='data/val2014', annotations='data/annotations/captions_val2014.json', transform=transform)
# rebuilds vocabulary if necessary or specified
# otherwise, uses the already prebuilt vocabulary
print("rebuilding vocabulary" if build_vocab == True else "using old vocabulary", file=sys.stderr)
word_to_index, index_to_word  = data_loader.create_vocab(train_set, min_occurrence=10) if build_vocab == True else (data_loader.load_vocab())
# overwrites the prebuilt vocabulary if specified, otherwise stores the vocabulary
if build_vocab == True:
  data_loader.write_vocab_to_file(index_to_word)

batch_size = 64
# batch the data
batched_train_set = data_loader.load_batched_data('batched_64_train_set_10.txt')
batched_val_set = data_loader.load_batched_data('batched_64_val_set_10.txt')
if batched_train_set == None:
  batched_train_set = data_loader.batch_data(train_set, word_to_index, batch_size=batch_size)
  data_loader.write_batched_data(batched_train_set, file_name="batched_64_train_set_10.txt")
if batched_val_set == None:
  batched_val_set = data_loader.batch_data(val_set, word_to_index, batch_size=batch_size)
  data_loader.write_batched_data(batched_val_set, file_name="batched_64_val_set_10.txt")

# creating the model
D_embed, H = 128, 256

encoder_cnn = EncoderCNN(D_embed)
model = LSTM(D_embed, H, len(word_to_index), batch_size)
if torch.cuda.is_available():
  model.cuda()
loss_function = nn.NLLLoss()
# weight decay parameter adds L2
optimizer = optim.Adam(model.parameters(), lr=0.0001)

record_error = False
if len(sys.argv) > 1:
  record_error = True

train_file = open(sys.argv[1], 'w') if record_error == True else None
val_file = open(sys.argv[2], 'w') if record_error == True else None

'''
## Training Design
## epoch is the number of times we go through the training set
## Every 1000 training batches we compute the average of 100 validate batches
## Every time we finish the training set we compute the average of the validate set
'''
index = 0
for epoch in range(10):
  train_keys = batched_train_set.keys()
  random.shuffle(train_keys)
  val_keys = batched_val_set.keys()
  random.shuffle(val_keys)
  data_set = []
  for train_key in train_keys:
    train_key_set = batched_train_set[train_key]
    for i in range(0, (len(train_key_set) / batch_size) - 1):
      image_caption_set = train_key_set[i * batch_size : (i + 1) * batch_size]
      data_set.append(image_caption_set)
  random.shuffle(data_set)
  train_sum_loss = 0
  for image_caption_set in data_set:
    index += 1
    model.train()
    loss = trainer.train_model(encoder_cnn, model, loss_function, optimizer, image_caption_set, train_set)
    train_sum_loss += loss
    if index % 100 == 0:
      if record_error == True:
        train_file.write(str(index) + "," + str(train_sum_loss / 100) + "\n")
      train_sum_loss = 0
    if index % 1000 == 0:
      val_sum_loss = 0
      model.eval()
      for i in range(100):
        val_sum_loss += trainer.eval_model_random(encoder_cnn, model, loss_function, val_set, batched_val_set, word_to_index, batch_size=batch_size)
      if record_error == True:
        val_sum_loss = val_sum_loss / 100
        val_file.write(str(index) + "," + str(val_sum_loss) + "\n")
  val_sum_loss = 0
  num_trials = 0
  if record_error == True:
    val_file.write("End of Epoch \n")
  for val_key in val_keys:
    val_key_set = batched_val_set[val_key]
    random.shuffle(val_key_set)
    model.eval()
    for i in range(0, (len(val_key_set) / batch_size) - 1):
      image_caption_set = val_key_set[i * batch_size : (i + 1) * batch_size]
      loss = trainer.eval_model(encoder_cnn, model, loss_function, image_caption_set, val_set)
      val_sum_loss += loss
      num_trials += 1
  if record_error == True:
    val_file.write(str(index) + "," + str(val_sum_loss / num_trials) + "\n")

if record_error == True:
  train_file.close()
  val_file.close()
torch.save(model.state_dict(), 'model/model_10epoch_dropout_3_0001.pt')
