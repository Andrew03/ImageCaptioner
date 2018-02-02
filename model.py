import torch
import torchvision.models as models
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
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
    features = features.view(features.size(0), -1).unsqueeze(0)
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
    #self.image_embedding_layer = nn.Linear(4096, embedding_dim)
    self.image_embedding_layer = nn.Linear(2048, embedding_dim)
    self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2)
    self.dropout_layer = nn.Dropout(p=dropout)
    self.hidden2word = nn.Linear(hidden_dim, vocab_size)
    self.hidden = self.init_hidden()

  def copy(self):
    decoder_copy = DecoderRNN(self.embedding_dim, self.hidden_dim, self.vocab_size, self.batch_size, self.dropout, self.useCuda)
    decoder_copy.word_embedding_layer = copy.deepcopy(self.word_embedding_layer)
    decoder_copy.image_embedding_layer = copy.deepcopy(self.image_embedding_layer)
    decoder_copy.batch_norm = copy.deepcopy(self.batch_norm)
    decoder_copy.lstm = copy.deepcopy(self.lstm)
    decoder_copy.dropout_layer = copy.deepcopy(self.dropout_layer)
    decoder_copy.hidden2word = copy.deepcopy(self.hidden2word)
    decoder_copy.hidden = decoder_copy.init_hidden()
    return decoder_copy

  def init_hidden(self, batch_size=None):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() and self.useCuda else torch.FloatTensor
    if batch_size:
      return (autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim)).type(dtype),
        autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim)).type(dtype))
    else:
      return (autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).type(dtype),
        autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).type(dtype))

  def forward(self, batch):
    is_word = type(batch.data) == torch.cuda.LongTensor or type(batch.data) == torch.LongTensor
    embed_value = self.word_embedding_layer(batch) if is_word else self.image_embedding_layer(batch)
    if not is_word:
      embed_value = self.batch_norm(embed_value.squeeze(0)).unsqueeze(0)
    #print(embed_value.norm().data[0])
    embed_value = self.dropout_layer(embed_value)
    self.lstm.flatten_parameters()
    lstm_out, self.hidden = self.lstm(
      embed_value.view(len(batch[0]), len(batch), self.embedding_dim), self.hidden) if is_word else self.lstm(embed_value, self.hidden)
    #print(self.hidden[0].norm().data[0])
    lstm_out = self.dropout_layer(lstm_out)
    words_space = self.hidden2word(lstm_out.view(-1, self.hidden_dim))
    return F.log_softmax(words_space, dim=1), F.softmax(words_space, dim=1)
