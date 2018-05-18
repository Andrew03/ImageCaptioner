import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import sys

def to_var(x, useCuda=True, volatile=False):
  if torch.cuda.is_available() and useCuda:
    x = x.cuda()
  return autograd.Variable(x, volatile=volatile)

def train(encoder_cnn, decoder_rnn, loss_function, optimizer, images, captions, grad_clip, useCuda):
  encoder_cnn.eval()
  decoder_rnn.train()
  decoder_rnn.zero_grad()
  #decoder_rnn.hidden = decoder_rnn.init_hidden()

  input_images = to_var(images, useCuda, volatile=True)
  # stripping away the <EOS> token from inputs
  len_caption = len(captions[0])
  input_captions = to_var(torch.index_select(captions, 1, torch.LongTensor([i for i in range(len_caption - 1)])), useCuda)
  # stripping away the <SOS> token from targets
  #targets = to_var(torch.index_select(captions, 1, torch.LongTensor([i for i in range(1, len_caption)])), useCuda)
  targets = to_var(captions, useCuda)
  #targets = to_var(torch.index_select(captions, 1, torch.LongTensor([i for i in range(len_caption - 1)])), useCuda)
  target_captions = pack_padded_sequence(targets, [len_caption for i in range(len(captions))], batch_first=True)[0]
  lengths = [len_caption for _ in range(len(captions))]

  # initializing decoder hidden state with image
  """
  decoder_rnn(autograd.Variable(encoder_cnn(input_images).data))
  caption_scores, _ = decoder_rnn(input_captions)
  """
  features = autograd.Variable(encoder_cnn(input_images).data)
  caption_scores, _ = decoder_rnn(features, input_captions, lengths)
  loss = loss_function(caption_scores, target_captions)
  loss.backward()
  nn.utils.clip_grad_norm(decoder_rnn.parameters(), grad_clip)
  optimizer.step()
  return loss.data.select(0, 0)
