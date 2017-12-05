from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import os.path
import data_loader
import file_namer
import param_parser
import evaluator
from nltk import bleu
from lstm import LSTM
from encoder import EncoderCNN

'''
## Run Parameters
## Grabbing the run parameters from the parameters file
'''
params = param_parser.parse_run_params(sys.argv[1])
if params is None:
  print("invalid run parameter file")
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
num_runs = params[9]
isNormalized = params[10]

# defining image size
transform = transforms.Compose([
  transforms.Scale(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  # PyTorch says iamges must be normalized like this
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

'''
## Loading the different data sets the model uses to evaluate 
## val_set stores the actual images and captions in word form
## word_to_index and index_to_word are used to store the vocabulary
## batched_val_set stores the batched data sets of image_indices and captions
'''
# loading the data sets
val_set = data_loader.load_data(images='data/val2014', annotations='data/annotations/captions_val2014.json', transform=transform)
# loads the vocabulary
word_to_index, index_to_word = data_loader.load_vocab(file_namer.make_vocab_name(min_occurrences))
# loads the batched data
batched_val_set = data_loader.load_batched_data(file_namer.make_batch_name(batch_size, min_occurrences, isTrain=False))

'''
## Creating the Model
## Instantiate the encoder and lstm and loss function
'''
encoder_cnn = EncoderCNN(isNormalized)
model = LSTM(embedding_dim, hidden_size, len(word_to_index), batch_size, dropout=dropout)
if torch.cuda.is_available():
  model.cuda()
loss_function = nn.NLLLoss()

'''
## Loading Checkpoint
## Loads the model from the checkpoint specified in run parameters
'''
checkpoint_name = file_namer.make_save_name(batch_size, min_occurrences, num_epochs, dropout, \
  model_lr, encoder_lr, embedding_dim, hidden_size, grad_clip, isNormalized=isNormalized)
if not os.path.isfile(checkpoint_name):
  print("Invalid run parameters!")
  print("Checkpoint " + str(checkpoint_name) + " does not exist!")
  sys.exit()
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
#model.load_state_dict(torch.load(checkpoint_name))

initial_word = ""
model.eval()
for epoch in range(num_runs):
  model.hidden = model.init_hidden()
  image, image_index, captions = evaluator.create_predict_batch(val_set, batched_val_set)
  prediction = evaluator.beam_search(encoder_cnn, model, image, beam_size=10)
  for caption in prediction:
    print("score is: " + str(caption[0]) + ", caption is: " + data_loader.caption_to_string(caption[1], index_to_word))
  print("actual captions are")
  captions = data_loader.get_all_captions(image_index, batched_val_set)
  for caption in captions:
    print(data_loader.caption_to_string(caption, index_to_word))
