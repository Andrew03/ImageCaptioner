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
import model
#from nltk import bleu

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
decoder_lr = params[5]
encoder_lr = params[6]
num_epochs = params[7]
grad_clip = params[8]
num_runs = params[9]
beam_size = params[10]
printStepProb = params[11]
isNormalized = params[12]
useCuda = params[13] and torch.cuda.is_available()

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
## Loading the different data sets the decoder_rnn uses to evaluate 
## val_set stores the actual images and captions in word form
## word_to_index and index_to_word are used to store the vocabulary
## batched_val_set stores the batched data sets of image_indices and captions
'''
# loading the data sets
#val_set = data_loader.load_data(images='data/val2014', annotations='data/annotations/captions_val2014.json', transform=transform)
val_set = data_loader.load_data(images='data/train2014', annotations='data/annotations/captions_train2014.json', transform=transform)
# loads the vocabulary
word_to_index, index_to_word = data_loader.load_vocab(file_namer.make_vocab_name(min_occurrences))
# loads the batched data
#batched_val_set = data_loader.load_batched_data(file_namer.make_batch_name(batch_size, min_occurrences, isTrain=False))
batched_val_set = data_loader.load_batched_data(file_namer.make_batch_name(batch_size, min_occurrences, isTrain=True))

'''
## Creating the decoder_rnn
## Instantiate the encoder and lstm and loss function
'''
encoder_cnn = model.EncoderCNN(isNormalized, useCuda)
decoder_rnn = model.DecoderRNN(embedding_dim, hidden_size, len(word_to_index), batch_size, dropout, useCuda)
if useCuda:
  decoder_rnn.cuda()
loss_function = nn.NLLLoss()

'''
## Loading Checkpoint
## Loads the decoder_rnn from the checkpoint specified in run parameters
'''
checkpoint_name = file_namer.make_checkpoint_name(batch_size, min_occurrences, num_epochs, dropout, \
  decoder_lr, encoder_lr, embedding_dim, hidden_size, grad_clip, isNormalized=isNormalized)
if not os.path.isfile(checkpoint_name):
  print("Invalid run parameters!")
  print("Checkpoint " + str(checkpoint_name) + " does not exist!")
  sys.exit()
checkpoint = torch.load(checkpoint_name) if useCuda else torch.load(checkpoint_name, map_location=lambda storage, loc: storage)
decoder_rnn.load_state_dict(checkpoint['state_dict'])

#sys.exit()
for epoch in range(num_runs):
  image, image_index, captions = evaluator.create_predict_batch(val_set, batched_val_set, useCuda)
  prediction = evaluator.beam_search(encoder_cnn, decoder_rnn, image, beam_size, useCuda, printStepProb)
  for caption in prediction:
    print("score is: " + str(caption[0]) + ", caption is: " + data_loader.caption_to_string(caption[1], index_to_word))
  print("actual captions are")
  captions = data_loader.get_all_captions(image_index, batched_val_set)
  for caption in captions:
    print(data_loader.caption_to_string(caption, index_to_word))
