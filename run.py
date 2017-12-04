from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import data_loader
import file_namer
from nltk import bleu
from lstm import LSTM
from encoder import EncoderCNN
import evaluator

# defining image size
transform = transforms.Compose([
  transforms.Scale(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

val_set = data_loader.load_data(images='data/val2014', annotations='data/annotations/captions_val2014.json', transform=transform)

min_occurrences, batch_size = 10, 64
word_to_index, index_to_word = data_loader.load_vocab(file_namer.make_vocab_name(min_occurrences))
batched_val_set = data_loader.load_batched_data(file_namer.make_batch_name(batch_size, min_occurrences, isTrain=False))

D_embed, H = 256, 256

encoder_cnn = EncoderCNN()
model = LSTM(D_embed, H, len(word_to_index), batch_size)
if torch.cuda.is_available():
  model.cuda()

model.load_state_dict(torch.load(sys.argv[1]))
loss_function = nn.NLLLoss()
initial_word = ""
model.eval()
for epoch in range(5):
  model.hidden = model.init_hidden()
  image, captions = evaluator.create_predict_batch(val_set, batched_val_set)
  prediction = evaluator.beam_search(encoder_cnn, model, image, beam_size=10)
  for caption in prediction:
    print("score is: " + str(caption[0]) + ", caption is: " + data_loader.caption_to_string(caption[1], index_to_word))
  print("actual caption is: " + data_loader.caption_to_string(captions, index_to_word))
