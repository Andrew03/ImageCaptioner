import torch
import torchvision.models as models
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
  def __init__(self, isNormalized=False, useCuda=True):
    super(EncoderCNN, self).__init__()
    self.vgg16 = models.vgg16_bn(pretrained=True) if isNormalized else models.vgg16(pretrained=True)
    # gets rid of dropout
    self.vgg16.eval()
    if torch.cuda.is_available() and useCuda:
      self.vgg16.cuda()
    # removing clasification layer
    del(self.vgg16.classifier._modules['6'])
    for param in self.vgg16.parameters():
      param.requires_grad = False

  def forward(self, images):
    return self.vgg16(images).unsqueeze(0)

class DecoderRNN(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size, dropout=0, useCuda=True):
    super(DecoderRNN, self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    self.image_embedding_layer = nn.Linear(4096, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    self.dropout = nn.Dropout(p=dropout)
    self.embedding_dim = embedding_dim
    self.hidden2word = nn.Linear(hidden_dim, vocab_size)
    self.batch_size = batch_size
    self.useCuda = useCuda
    self.hidden = self.init_hidden()

  def init_hidden(self, batch_size=None):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() and self.useCuda else torch.FloatTensor
    if batch_size:
      return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).type(dtype),
        autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).type(dtype))
    else:
      return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).type(dtype),
        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).type(dtype))

  def forward(self, batch):
    is_word = type(batch.data) == torch.cuda.LongTensor or type(batch.data) == torch.LongTensor
    embed_value = self.word_embedding_layer(batch) if is_word else self.image_embedding_layer(batch)
    embed_value = self.dropout(embed_value)
    lstm_out, self.hidden = self.lstm(
      embed_value.view(len(batch[0]), len(batch), self.embedding_dim), self.hidden) if is_word else self.lstm(embed_value)
    lstm_out = self.dropout(lstm_out)
    words_space = self.hidden2word(lstm_out.view(-1, self.hidden_dim))
    return F.log_softmax(words_space, dim=1), F.softmax(words_space, dim=1)
