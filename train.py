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
word_to_index, index_to_word  = data_loader.create_vocab(train_set, min_occurrence=5) if build_vocab == True else (data_loader.load_vocab())
# overwrites the prebuilt vocabulary if specified, otherwise stores the vocabulary
if build_vocab == True:
  data_loader.write_vocab_to_file(index_to_word)

# batch the data
batched_train_set = data_loader.load_batched_data('batched_train_set.txt', word_to_index)
batched_val_set = data_loader.load_batched_data('batched_val_set.txt', word_to_index)
if batched_train_set == None:
  batched_train_set = data_loader.batch_data(train_set, word_to_index, batch_size=32)
  data_loader.write_batched_data(batched_train_set, file_name="batched_train_set.txt")
if batched_val_set == None:
  batched_val_set = data_loader.batch_data(val_set, word_to_index, batch_size=32)
  data_loader.write_batched_data(batched_val_set, file_name="batched_val_set.txt")

# creating the model
batch_size, min_occurrences = 32, 10
D_embed, H = 128, 256

encoder_cnn = EncoderCNN(D_embed)
model = LSTM(D_embed, H, len(word_to_index), batch_size)
if torch.cuda.is_available():
  model.cuda()
loss_function = nn.NLLLoss()
# weight decay parameter adds L2
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

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
for epoch in range(5):
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
  for image_caption_set in data_set:
    index += 1
    model.train()
    loss = trainer.train_model(encoder_cnn, model, loss_function, optimizer, image_caption_set, train_set)
    if record_error == True:
      train_file.write(str(index) + "," + str(loss) + "\n")
    if index % 1000 == 0:
      sum_loss = 0
      model.eval()
      for i in range(100):
        sum_loss += trainer.eval_model_random(encoder_cnn, model, loss_function, val_set, batched_val_set, word_to_index, batch_size=batch_size)
      if record_error == True:
        sum_loss = sum_loss / 100
        val_file.write(str(index) + "," + str(sum_loss) + "\n")
  sum_loss = 0
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
      sum_loss += loss
      num_trials += 1
  if record_error == True:
    val_file.write(str(index) + "," + str(sum_loss / num_trials) + "\n")

if record_error == True:
  train_file.close()
  val_file.close()
torch.save(model.state_dict(), 'model/model_5epoch_dropout_2.pt')
