import torch
import torch.autograd as autograd
import torchvision.datasets as datasets
import os.path
import re
import json
import random
import sys
import string
from random import randint

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

# input the vocab that contains the words, i.e index_to_word
def write_vocab_to_file(vocab):
  vocab_file = open('vocab.txt', 'w')
  for word in vocab:
    vocab_file.write(word + "\n")
  vocab_file.close()

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

def load_batched_data(file_name, word_to_index):
  if os.path.isfile(file_name) == True:
    batched_data = {}
    data_file = open(file_name, 'r')
    num_keys = int(data_file.readline())
    for _ in range(num_keys):
      key = int(data_file.readline())
      num_caps = int(data_file.readline())
      batched_data[key] = []
      for _ in range(num_caps):
        image = int(data_file.readline())
        caption = data_file.readline().split(',')
        caption = [int(i) for i in caption[:-1]]
        batched_data[key].append([image, caption])
    return batched_data
  else:
    return None

def write_batched_data(batched_data, file_name):
  data_file = open(file_name, 'w')
  data_file.write(str(len(batched_data)) + "\n")
  for key in batched_data.keys():
    data_file.write(str(key) + "\n")
    data_file.write(str(len(batched_data[key])) + "\n")
    for image, sentence in batched_data[key]:
      data_file.write(str(image) + "\n")
      for word_index in sentence:
        data_file.write(str(word_index) + ",")
      data_file.write("\n")
  data_file.close()

def batch_data(data_set, word_to_index, batch_size=1):
  batched_set = {}
  for i in range(len(data_set)):
    image, captions = data_set[i]
    for caption in captions:
      sentence = split_sentence(caption)
      sentence.insert(0, "SOS")
      sentence.append("EOS")
      if len(sentence) not in batched_set.keys():
        batched_set[len(sentence)] = []
      batched_set[len(sentence)].append([i, [get_index(word, word_to_index) for word in sentence]])
  for i in batched_set.keys():
    if len(batched_set[i]) < batch_size:
      del batched_set[i]
  return batched_set

def load_data(images, annotations, transform):
  return datasets.CocoCaptions(root = images, annFile = annotations, transform = transform)

# Needs an array of images
def image_to_variable(image):
  if torch.cuda.is_available():
    image = image.cuda()
  return autograd.Variable(image)

# returns images in a stored tensor, captions are just in a list, need to format to input or output manually
def create_batch(image_set, batched_data, word_to_index, batch_size=1, randomize=False):
  images = []
  captions = []
  index = random.choice(batched_data.keys())
  data_set = batched_data[index]
  random.shuffle(data_set)
  image_caption_set = data_set[0:32]
  for image_caption in image_caption_set:
    image, _ = image_set[image_caption[0]]
    images.append(image)
    captions.append(image_caption[1])
  images = image_to_variable(torch.stack(images, 0))
  return images, captions

def create_input_batch_images(image_set, image_indices):
  images = [image_set[i][0] for i in image_indices]
  return image_to_variable(torch.stack(images, 0)) 

def create_input_batch_captions(captions):
  inputs = []
  for caption in captions:
    # strip the EOS token if it is there
    if len(caption) > 1:
      inputs.append(caption[:-1])
    else:
      inputs.append(caption)
  return autograd.Variable(torch.cuda.LongTensor(inputs)) if torch.cuda.is_available() else autograd.Variable(torch.LongTensor(inputs))

# targets are a long vector, flatten them out
# remember to take SOS token from targets
def create_target_batch_captions(captions):
  targets = []
  for caption in captions: 
    for word_index in caption[1:]:
      targets.append(word_index)
  return autograd.Variable(torch.cuda.LongTensor(targets)) if torch.cuda.is_available() else autograd.Variable(torch.LongTensor(targets))

def get_index(word, word_to_index):
  return word_to_index[word] if word in word_to_index else word_to_index["UNK"]

def split_sentence(sentence):
  remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
  sentence = sentence.translate(remove_punctuation_map)
  return re.findall(r"[\w']+|[.,!?;]", sentence.lower())

def caption_to_string(caption, index_to_word):
  string_rep = ""
  for word in caption:
    if word != 1 and word != 2:
      string_rep += index_to_word[word] + " "
  return string_rep[:-1]
