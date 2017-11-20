from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.autograd as autograd
import sys
import data_loader
from nltk import bleu
from lstm import LSTM
from encoder import EncoderCNN
import evaluator
import heapq
import operator

# defining image size
transform = transforms.Compose([
  transforms.Scale(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

training_set = data_loader.load_data(images='data/val2014', annotations='data/annotations/captions_val2014.json', transform=transform)

print("using old vocabulary")
word_to_index, index_to_word = data_loader.load_vocab()

batched_val_set = data_loader.load_batched_data('batched_val_set_10.txt')

batch_size = 32
D_embed, H = 128, 256

encoder_cnn = EncoderCNN(D_embed)
model = LSTM(D_embed, H, len(word_to_index), batch_size)
if torch.cuda.is_available():
  model.cuda()

model.load_state_dict(torch.load(sys.argv[1]))
loss_function = nn.NLLLoss()
initial_word = ""
model.eval()
for epoch in range(5):
  model.hidden = model.init_hidden()
  image, captions = evaluator.create_predict_batch(training_set, batched_val_set)


  prediction = evaluator.beam_search(encoder_cnn, model, image, beam_size=10)
  for caption in prediction:
    print("score is: " + str(caption[0]) + ", caption is: " + data_loader.caption_to_string(caption[1], index_to_word))
  #print(prediction)
  print("actual caption is: " + data_loader.caption_to_string(captions, index_to_word))
