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
from tqdm import tqdm
import data_loader
import file_namer
import param_parser
import trainer
from lstm import LSTM
from encoder import EncoderCNN

'''
## Training Parameters
## Grabbing the training parameters from the parameters file
'''
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
val_set = data_loader.load_data(images='data/val2014', annotations='data/annotations/captions_val2014.json', transform=transform)
# loads the vocabulary 
word_to_index, index_to_word = data_loader.load_vocab(file_namer.make_vocab_name(min_occurrences))
if word_to_index is None:
  print("run the setup script to create the vocab")
  sys.exit()
# loads the batched data
batched_train_set = data_loader.load_batched_data(file_namer.make_batch_name(batch_size, min_occurrences, isTrain=True))
batched_val_set = data_loader.load_batched_data(file_namer.make_batch_name(batch_size, min_occurrences, isTrain=False))
if batched_train_set is None or batched_val_set is None:
  print("run the setup script to batch the data")
  sys.exit()

'''
## Creating the Model and Optimizer
## Instantiate the encoder and lstm and loss function and optimizer
'''
encoder_cnn = EncoderCNN(isNormalized)
model = LSTM(embedding_dim, hidden_size, len(word_to_index), batch_size, dropout=dropout)
if torch.cuda.is_available():
  model.cuda()
loss_function = nn.NLLLoss()
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
output_val_file_name = file_namer.make_output_name(batch_size, min_occurrences, \
  num_epochs, dropout, model_lr, encoder_lr, embedding_dim, hidden_size, \
  grad_clip, isTrain=False, isNormalized=isNormalized)
output_val_file = open(output_val_file_name, 'w')

index = 0
start_epoch = 0

'''
## Loading Checkpoint
## Searches for a saved checkpoint, using the one with the most epochs if one exists
## Loads the epoch, index, model and optimizer states
'''
save_name = file_namer.make_save_name(batch_size, min_occurrences, num_epochs, dropout, \
  model_lr, encoder_lr, embedding_dim, hidden_size, grad_clip, isNormalized=isNormalized)
checkpoint_name = file_namer.get_checkpoint(save_name)
if checkpoint_name is not None:
  print("loading from checkpoint " + str(checkpiont_name))
  checkpoint = torch.load(checkpoint_name)
  start_epoch = checkpoint['epoch']
  index = checkpoint['index']
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  
'''
## Training Design
## epoch is the number of times we go through the training set
## Every 1000 training batches we compute the average of 100 validate batches
## Every time we finish the training set we compute the average of the validate set
'''
data_set = data_loader.shuffle_data_set(batched_train_set, batch_size)
with tqdm(total=len(data_set) * num_epochs + 1) as progress_bar:
  progress_bar.set_description('Epoch %i (Train)' % start_epoch)
  for epoch in range(start_epoch, num_epochs):
    train_sum_loss = 0
    for image_caption_set in data_set:
      index += 1
      model.train()
      train_sum_loss += trainer.train_model(encoder_cnn, model, loss_function, optimizer, image_caption_set, train_set, grad_clip)
      # record loss
      if index % 100 == 0:
        output_train_file.write(str(index) + "," + str(train_sum_loss / 100) + "\n")
        progress_bar.set_postfix(epoch=epoch, loss=(train_sum_loss / 100), index=index)
        progress_bar.update(100)
        train_sum_loss = 0
        # run a random validation sample
        if index % 1000 == 0:
          val_sum_loss = 0
          model.eval()
          for i in range(100):
            val_sum_loss += trainer.eval_model_random(encoder_cnn, model, loss_function, val_set, batched_val_set, word_to_index, batch_size=batch_size)
          output_val_file.write(str(index) + "," + str(val_sum_loss / 100) + "\n")
    tqdm.write("rebatching for epoch " + str(epoch))
    if epoch != num_epochs + 1:
      data_set = data_loader.shuffle_data_set(batched_train_set, batch_size)

    # run an entire validation batch
    tqdm.write("validating epoch " + str(epoch))
    val_sum_loss = 0
    num_trials = 0
    output_val_file.write("End of Epoch \n")
    val_keys = batched_val_set.keys()
    for val_key in random.sample(val_keys, len(val_keys)):
      val_key_set = random.sample(batched_val_set[val_key], len(batched_val_set[val_key]))
      model.eval()
      data_set = data_loader.group_data(val_key_set, batch_size)
      for image_caption_set in data_set:
        loss = trainer.eval_model(encoder_cnn, model, loss_function, image_caption_set, val_set)
        val_sum_loss += loss
        num_trials += 1
        if num_trials % 100 == 0:
          progress_bar.set_description('Epoch %i (Val)' % epoch)
          progress_bar.set_postfix(loss=(val_sum_loss / num_trials), trial_number=num_trials, out_of=len(data_set))
    output_val_file.write(str(index) + "," + str(val_sum_loss / num_trials) + "\n")

output_train_file.close()
output_val_file.close()
'''
torch.save(model.state_dict(), file_namer.make_save_name(batch_size, \
  min_occurrences, num_epochs, dropout, model_lr, encoder_lr, embedding_dim, \
  hidden_size, grad_clip, isNormalized=isNormalized))
'''
torch.save({'epoch': num_epoch,
            'index': index,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}, file_namer.make_save_name(batch_size, \
              min_occurrences, num_epochs, dropout, model_lr, encoder_lr, \
              embedding_dim, hidden_size, grad_clip, isNormalized=isNormalized))
