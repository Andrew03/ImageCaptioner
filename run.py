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

batched_val_set = data_loader.load_batched_data('batched_val_set.txt', word_to_index)

batch_size, min_occurrences = 32, 10
D_embed, H, D_out = 128, 256, 32

encoder_cnn = EncoderCNN(D_embed)
model = LSTM(D_embed, H, len(word_to_index), batch_size)
if torch.cuda.is_available():
    model.cuda()

model.load_state_dict(torch.load('model/model_1epoch_dropout_2.pt'))
loss_function = nn.NLLLoss()
initial_word = ""
model = model.eval()
for epoch in range(1):
    model.hidden = model.init_hidden()
    image, captions = evaluator.create_predict_batch(training_set, batched_val_set, batch_size=32)
    prediction = evaluator.beam_search(encoder_cnn, model, image, 32, beam_size=10)
    for caption in prediction:
      print("score is: " + str(caption[0]) + ", caption is: " + data_loader.caption_to_string(caption[1], index_to_word))
    #print(prediction)
    print(captions[0])
