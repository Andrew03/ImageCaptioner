import torch
import torchvision.models as models
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import copy

class EncoderCNN(nn.Module):
  def __init__(self, isNormalized=False, useCuda=True):
    super(EncoderCNN, self).__init__()
    #self.vgg16 = models.vgg16_bn(pretrained=True) if isNormalized else models.vgg16(pretrained=True)
    resnet = models.resnet152(pretrained=True)
    modules = list(resnet.children())[:-1]      # delete the last fc layer.
    self.resnet = nn.Sequential(*modules)
    # gets rid of dropout
    #self.vgg16.eval()
    self.resnet.eval()
    if torch.cuda.is_available() and useCuda:
      #self.vgg16.cuda()
      self.resnet.cuda()
    # removing clasification layer
    """
    del(self.vgg16.classifier._modules['6'])
    for param in self.vgg16.parameters():
      param.requires_grad = False
    """
    for param in self.resnet.parameters():
      param.requires_grad = False

  def forward(self, images):
    #return self.vgg16(images).unsqueeze(0)
    features = autograd.Variable(self.resnet(images).data)
    features = features.view(features.size(0), -1)
    return features

class DecoderRNN(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size, dropout=0, useCuda=True):
    super(DecoderRNN, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.dropout = dropout
    self.useCuda = useCuda

    self.word_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    self.image_embedding_layer = nn.Linear(2048, embedding_dim)
    self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True)
    self.dropout_layer = nn.Dropout(p=dropout)
    self.hidden2word = nn.Linear(hidden_dim, vocab_size)

  def copy(self):
    decoder_copy = DecoderRNN(self.embedding_dim, self.hidden_dim, self.vocab_size, self.batch_size, self.dropout, self.useCuda)
    decoder_copy.word_embedding_layer = copy.deepcopy(self.word_embedding_layer)
    decoder_copy.image_embedding_layer = copy.deepcopy(self.image_embedding_layer)
    decoder_copy.batch_norm = copy.deepcopy(self.batch_norm)
    decoder_copy.lstm = copy.deepcopy(self.lstm)
    decoder_copy.dropout_layer = copy.deepcopy(self.dropout_layer)
    decoder_copy.hidden2word = copy.deepcopy(self.hidden2word)
    return decoder_copy

  def forward(self, images, captions, lengths):
    image_features = self.image_embedding_layer(images)
    word_embeddings = self.word_embedding_layer(captions)
    embeddings = torch.cat((image_features.unsqueeze(1), word_embeddings), 1)
    packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
    self.lstm.flatten_parameters()
    hiddens, _ = self.lstm(packed)
    outputs = self.hidden2word(hiddens[0])
    return F.log_softmax(outputs, dim=1), F.softmax(outputs, dim=1)
