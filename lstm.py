import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        #self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.embedding_dim = embedding_dim
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).type(torch.cuda.FloatTensor),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).type(torch.cuda.FloatTensor)) if torch.cuda.is_available() else (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).type(torch.FloatTensor),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).type(torch.FloatTensor))

    def forward(self, sentence):
        #embed_value = self.embedding_layer(sentence)
        lstm_out, self.hidden = self.lstm(
            sentence.view(-1, len(sentence), self.embedding_dim), self.hidden )
        words_space = self.hidden2word(lstm_out.view(-1, self.hidden_dim))
        words_score = F.log_softmax(words_space)
        return words_score, words_space
