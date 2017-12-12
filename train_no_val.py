from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as autograd
import torchvision.models as models
import sys
import random
from tqdm import tqdm
import data_loader
import file_namer
import param_parser
import trainer
import model

'''
## Training Parameters
## Grabbing the training parameters from the parameters file
'''
if len(sys.argv) != 2:
  print("need a training file")
  sys.exit()
params = param_parser.parse_train_params(sys.argv[1])
if params is None:
  print("invalid training parameter file")
  sys.exit()
min_occurrences = params[0]
batch_size = params[1]
embedding_dim = params[2]
hidden_size = params[3]
dropout = params[4]
model_lr = params[5]
encoder_lr = params[6]
num_epochs = params[7]
grad_clip = params[8]
isNormalized = params[9]
useCuda = params[10]

# defining image size
transform = transforms.Compose([
  transforms.Scale(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  # PyTorch says images must be normalized like this
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

'''
## Loading the different data sets the model uses to train
## train_set and val_set store the actual images and captions in word form
## word_to_index and index_to_word are used to store the vocabulary
## batched_train_set and batched_val_set store batched sets of image_indices and captions
'''
# loading the data sets
train_set = data_loader.load_data(images='data/train2014', annotations='data/annotations/captions_train2014.json', transform=transform)
# loads the vocabulary 
word_to_index, index_to_word = data_loader.load_vocab(file_namer.make_vocab_name(min_occurrences))
if word_to_index is None:
  print("run the setup script to create the vocab")
  sys.exit()
# loads the batched data
batched_train_set = data_loader.load_batched_data(file_namer.make_batch_name(batch_size, min_occurrences, isTrain=True))
if batched_train_set is None:
  print("run the setup script to batch the data")
  sys.exit()

'''
## Creating the Model and Optimizer
## Instantiate the encoder and lstm and loss function and optimizer
'''
encoder_cnn = model.EncoderCNN(isNormalized, useCuda=useCuda)
model = model.DecoderRNN(embedding_dim, hidden_size, len(word_to_index), batch_size, dropout=dropout, useCuda=useCuda)
if torch.cuda.is_available() and useCuda:
  model.cuda()
loss_function = nn.NLLLoss()
sys.exit()
# weight decay parameter adds L2
optimizer = optim.Adam([
  {'params': model.word_embedding_layer.parameters()},
  {'params': model.lstm.parameters()},
  {'params': model.hidden2word.parameters()},
  {'params': model.image_embedding_layer.parameters(), 'lr': encoder_lr},
  ], lr=model_lr)

'''
## Output Files
## Creating output files to write results to
'''
output_train_file_name = file_namer.make_output_name(batch_size, min_occurrences, \
  num_epochs, dropout, model_lr, encoder_lr, embedding_dim, hidden_size, \
  grad_clip, isTrain=True, isNormalized=isNormalized)
output_train_file = open(output_train_file_name, 'w')

index = 0
start_epoch = 0

'''
## Loading Checkpoint
## Searches for a saved checkpoint, using the one with the most epochs if one exists
## Loads the epoch, index, model and optimizer states
'''
save_name = file_namer.make_checkpoint_name(batch_size, min_occurrences, num_epochs, dropout, \
  model_lr, encoder_lr, embedding_dim, hidden_size, grad_clip, isNormalized=isNormalized)
checkpoint_name = file_namer.get_checkpoint(save_name)
checkpoint_name = None
for i in range(num_epochs, 0, -1):
  temp = file_namer.make_checkpoint_name(batch_size, min_occurrences, num_epochs, dropout, \
    model_lr, encoder_lr, embedding_dim, hidden_size, grad_clip, isNormalized=isNormalized)
  if file_namer.is_checkpoint(temp):
    checkpoint_name = temp
    print(checkpoint_name)
    break
'''
if checkpoint_name is not None:
  print("loading from checkpoint " + str(checkpoint_name))
  checkpoint = torch.load(checkpoint_name)
  start_epoch = checkpoint['epoch']
  index = checkpoint['index']
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
'''
  
'''
## Training Design
## epoch is the number of times we go through the training set
## Every 1000 training batches we compute the average of 100 validate batches
## Every time we finish the training set we compute the average of the validate set
'''
for epoch in range(start_epoch, num_epochs):
  train_data_set = data_loader.shuffle_data_set(batched_train_set, batch_size)
  train_sum_loss = 0
  progress_bar = tqdm(train_data_set)
  progress_bar.set_description('Epoch %i (Train)' % epoch)
  for image_caption_set in progress_bar:
    train_sum_loss += trainer.train_model(encoder_cnn, model, loss_function, optimizer, image_caption_set, train_set, grad_clip, useCuda)
    progress_bar.set_postfix(loss=train_sum_loss / ((index % 100) + 1))
    index += 1
    if index % 100 == 0:
      output_train_file.write(str(index) + "," + str(train_sum_loss / 100) + "\n")
      train_sum_loss = 0
  torch.save({'epoch': epoch + 1,
              'index': index,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()}, 'checkpoints/test_' + str(epoch + 1) + '.pt')

output_train_file.close()
