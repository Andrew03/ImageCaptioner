import torch
import torch.autograd as autograd
import torchvision.datasets as datasets
import os.path

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

def load_data(images, annotations, transform):
    return datasets.CocoCaptions(root = images, annFile = annotations, transform = transform)

# Needs an array of images
def image_to_variable(image):
    if torch.cuda.is_available():
        image = image.cuda()
    return autograd.Variable(image)

# assumes that data is in form of a two tuple, with captions as the second
def create_vocab(data, min_occurrence=1, unknown_val=0, end_of_seq_val=1, end_val=1):
    word_to_index = {}
    index_to_word = []
    word_to_appearences = {}
    word_to_index["UNK"] = unknown_val
    index_to_word.append("UNK")
    iter_number = 0
    for _, captions in data:
        if iter_number % 1000 == 0 and iter_number > 999:
            print iter_number
        for sentence in captions:
            for word in sentence.lower().split():
                if word not in word_to_appearences:
                    word_to_appearences[word] = 0
                word_to_appearences[word] += 1
                if word_to_appearences[word] == min_occurrence:
                    word_to_index[word] = len(index_to_word)
                    index_to_word.append(word)
        iter_number += 1
    return word_to_index, index_to_word
