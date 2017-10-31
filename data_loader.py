import torch
import torch.autograd as autograd
import torchvision.datasets as datasets
import os.path
import re
import json
import random
from random import randint

def load_image_information(path):
    json_data = {}
    with open(path) as f:
        json_data = json.load(f)
    return json_data['images']

def get_file_information():
    image_dir = ""
    annotation_dir = ""
    while os.path.isdir(image_dir) == False:
        print "image directory:",
        image_dir = raw_input()
    while os.path.isfile(annotation_dir) == False:
        print "annotation directory:",
        annotation_dir = raw_input()
    build_vocab = os.path.isfile("vocab.txt") == False
    if build_vocab == False:
        vocab_input = ""
        while vocab_input != "y" and vocab_input != "n":
            print "rebuild vocabulary? (y/n)",
            vocab_input = raw_input()
        build_vocab = (vocab_input == "y")
    return image_dir, annotation_dir, build_vocab

def split_sentence(sentence):
    return re.findall(r"[\w']+|[.,!?;]", sentence.lower())

# input the vocab that contains the words, i.e index_to_word
def write_vocab_to_file(vocab):
    vocab_file = open('vocab.txt', 'w')
    for word in vocab:
        vocab_file.write(word + "\n")
    vocab_file.close()

def load_vocab():
    vocab_file = open('vocab.txt', 'r')
    index_to_word = vocab_file.read().splitlines()
    vocab_file.close()
    word_to_index = {}
    index = 0
    for word in index_to_word:
        word_to_index[word] = index
        index += 1
    return word_to_index, index_to_word

def load_data(images, annotations, transform, batch_size=1):
    data_set =  datasets.CocoCaptions(root = images, annFile = annotations, transform = transform)
    training_set = {}
    if os.path.isfile('training_set.txt') == True:
        data_file = open('training_set.txt', 'r')
        num_keys = int(data_file.readline())
        for _ in range(0, num_keys):
            print ("index " + str(_) + " out of " + str(num_keys))
            key = int(data_file.readline())
            num_caps = int(data_file.readline())
            training_set[key] = []
            for _ in range(0, num_caps):
                image = int(data_file.readline())
                caption = data_file.readline()
                caption = caption[1:]
                caption = caption[:-1]
                caption = caption.split(", ")
                training_set[key].append([image, caption])
    for i in range(len(data_set)):
        image, captions = data_set[i]
        for caption in captions:
            sentence = split_sentence(caption)
            sentence.insert(0, "SOS")
            sentence.append("EOS")
            if len(sentence) not in training_set.keys():
                training_set[len(sentence)] = []
            training_set[len(sentence)].append([i, sentence])
    for i in training_set.keys():
        if len(training_set[i]) < batch_size:
            del training_set[i]
    data_file = open('training_set.txt', 'w')
    data_file.write(str(len(training_set))+ "\n")
    for key in training_set.keys():
        data_file.write(str(key)+ "\n")
        data_file.write(str(len(training_set[key])) + "\n")
        for image, sentence in training_set[key]:
            data_file.write(str(image) + "\n")
            data_file.write(str(sentence) + "\n")
    data_file.close()
    return data_set, training_set

def load_data_ungrouped(images, annotations, transform, batch_size=1):
    return datasets.CocoCaptions(root = images, annFile = annotations, transform = transform)

# Needs an array of images
def image_to_variable(image):
    if torch.cuda.is_available():
        image = image.cuda()
    return autograd.Variable(image)

# assumes that data is in form of a two tuple, with captions as the second
def create_vocab(data, min_occurrence=1, unknown_val=0, end_of_seq_val=1, start_of_seq_val=2):
    word_to_index = {}
    index_to_word = []
    word_to_appearences = {}
    word_to_index["UNK"] = unknown_val
    index_to_word.append("UNK")
    word_to_index["SOS"] = end_of_seq_val
    index_to_word.append("SOS")
    word_to_index["EOS"] = start_of_seq_val
    index_to_word.append("EOS")
    iter_number = 0
    for _, captions in data:
        if iter_number % 1000 == 0 and iter_number > 999:
            print iter_number
        for sentence in captions:
            for word in split_sentence(sentence):
                if word not in word_to_appearences:
                    word_to_appearences[word] = 0
                word_to_appearences[word] += 1
                if word_to_appearences[word] == min_occurrence:
                    word_to_index[word] = len(index_to_word)
                    index_to_word.append(word)
        iter_number += 1
    return word_to_index, index_to_word

def create_val_batch(training_set, word_to_index, num_captions, batch_size=1, randomize=False):
    images = []
    captions = []
    max_length = 0
    for _ in range(batch_size):
        randInt = randint(0, len(training_set) - 1)
        image, caption = training_set[randInt]
        images.append(image)
        sentence = split_sentence(caption[randint(0, num_captions - 1)])
        # inserting and appending start of string and end of string tags
        sentence.insert(0, "SOS")
        sentence.append("EOS")
        captions.append([get_index(word, word_to_index) for word in sentence])
        if len(sentence) > max_length:
            max_length = len(sentence)
    for _ in range(batch_size):
        while len(captions[_]) < max_length:
            captions[_].append(get_index("EOS", word_to_index))
    images = image_to_variable(torch.stack(images, 0))
    return images, captions

# returns images in a stored tensor, captions are just in a list, need to format to input or output manually
def create_batch(image_set, grouped_training_set, word_to_index, num_captions, batch_size=1, randomize=False):
    images = []
    captions = []
    training_set_index = random.choice(grouped_training_set.keys())
    data_set = grouped_training_set[training_set_index]
    random.shuffle(data_set)
    image_caption_set = data_set[0:32]
    for image_caption in image_caption_set:
        image, _ = image_set[image_caption[0]]
        images.append(image)
        captions.append([get_index(word, word_to_index) for word in image_caption[1]])
    images = image_to_variable(torch.stack(images, 0))
    return images, captions

def create_input_batch_captions(captions):
    return autograd.Variable(torch.cuda.LongTensor(captions)) if torch.cuda.is_available() else autograd.Variable(torch.LongTensor(captions))

# need to enlargen them probably to match vocabulary length
def create_input_batch_image_features(image_features, vocab_length):
    return image_features

# targets are a long vector, flatten them out
def create_target_batch_captions(captions):
    targets = []
    for caption in captions: 
        for word_index in caption:
            targets.append(word_index)
    return autograd.Variable(torch.cuda.LongTensor(targets)) if torch.cuda.is_available() else autograd.Variable(torch.LongTensor(targets))

def get_index(word, word_to_index):
    return word_to_index[word] if word in word_to_index else word_to_index["UNK"]
