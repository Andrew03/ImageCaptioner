import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size, dropout=0):
    super(LSTM,self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    self.image_embedding_layer = nn.Linear(4096, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    self.dropout = nn.Dropout(p=dropout)
    self.embedding_dim = embedding_dim
    self.hidden2word = nn.Linear(hidden_dim, vocab_size)
    self.batch_size = batch_size
    self.hidden = self.init_hidden()

  def init_hidden(self, batch_size=None):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
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
    # is this line correct? do i need to format hidden?
    lstm_out, self.hidden = self.lstm(
      embed_value.view(-1, len(batch), self.embedding_dim), self.hidden) if is_word else self.lstm(embed_value)
    lstm_out = self.dropout(lstm_out)
    words_space = self.hidden2word(lstm_out.view(-1, self.hidden_dim))
    words_score = F.log_softmax(words_space)
    return words_score, F.softmax(words_space)
