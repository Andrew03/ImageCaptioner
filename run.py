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

# defining image size
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

image_dir = ""
annotation_dir = ""
build_vocab = False
if len(sys.argv) == 1:
    image_dir, annotation_dir, build_vocab = data_loader.get_file_information()
elif len(sys.argv) < 3:
    print("Include image data set and caption data set")
    sys.exit()
else:
    image_dir = sys.argv[1]
    annotation_dir = sys.argv[2]

training_set = data_loader.load_data(images=image_dir, annotations=annotation_dir, transform=transform)

batch_size = 32
images = []

print("using old vocabulary")
word_to_index, index_to_word = data_loader.load_vocab()

batch_size, min_occurrences = 32, 10
D_embed, H, D_out = 32, 100, 30

encoder_cnn = EncoderCNN(D_embed)
model = LSTM(D_embed, H, len(word_to_index), batch_size)
feature_mapping = nn.Linear(4096, D_embed)
caption_embedding = nn.Embedding(len(word_to_index),D_embed)
if torch.cuda.is_available():
    model.cuda()
    feature_mapping.cuda()
    caption_embedding.cuda()
model.load_state_dict(torch.load('model/model2.pt'))
feature_mapping.load_state_dict(torch.load('model/feature_mapping2.pt'))
caption_embedding.load_state_dict(torch.load('model/caption_embedding2.pt'))
loss_function = nn.NLLLoss()
initial_word = ""
for epoch in range(1):
    model.zero_grad()
    model.hidden = model.init_hidden()

    images, captions = data_loader.create_val_batch(training_set, word_to_index, 5, batch_size=batch_size)
    target_caption = ""
    for word_index in captions[0]:
        if word_index == 1:
            break
        elif word_index != 2:
            target_caption += index_to_word[word_index] + " "

    image_features = encoder_cnn(images)
    #image_features = feature_mapping(image_features)
    #initial_score, initial_word_space = model(image_features)
    initial_score = model(image_features)
    sentence = ""
    index = 0
    input_batch = data_loader.create_input_batch_captions([[1,0] for _ in range(batch_size)])
    #input_features = caption_embedding(input_batch)
    #initial_score, initial_word_space = model(input_features)
    initial_score = model(input_batch)
    best_score, best_index = initial_score.data[0].max(0)
    best_word = index_to_word[best_index[0]]
    while index < 18 and best_word != "EOS":
        sentence += best_word + " "
        input_batch = data_loader.create_input_batch_captions([[best_index[0], 0] for _ in range(batch_size)])
        #input_features = caption_embedding(input_batch)
        #initial_score, initial_word_space = model(input_features)
        initial_score = model(input_batch)
        best_score, best_index = initial_score.data[0].max(0)
        best_word = index_to_word[best_index[0]]
        print(sentence)
        index += 1
    print(captions[0])
