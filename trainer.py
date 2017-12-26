import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import data_loader

def to_var(x, useCuda=True, volatile=False):
  if torch.cuda.is_available() and useCuda:
    x = x.cuda()
  return autograd.Variable(x, volatile=volatile)

def train(encoder_cnn, decoder_rnn, loss_function, optimizer, images, captions, lengths, grad_clip, useCuda):
  encoder_cnn.eval()
  decoder_rnn.train()
  decoder_rnn.zero_grad()
  decoder_rnn.hidden = decoder_rnn.init_hidden()

  input_images = to_var(images, useCuda, volatile=True)
  # stripping away the <EOS> token from inputs
  input_captions = to_var(torch.index_select(captions, 1, torch.LongTensor([i for i in range(lengths[0] - 1)])), useCuda)
  # stripping away the <SOS> token from targets
  targets = to_var(torch.index_select(captions, 1, torch.LongTensor([i for i in range(1, lengths[0])])), useCuda)
  target_captions = pack_padded_sequence(targets, lengths, batch_first=True)[0]

  # initializing decoder hidden state with image
  decoder_rnn(autograd.Variable(encoder_cnn(input_images).data))
  caption_scores, _ = decoder_rnn(input_captions)
  loss = loss_function(caption_scores, target_captions)
  loss.backward()
  nn.utils.clip_grad_norm(decoder_rnn.parameters(), grad_clip)
  optimizer.step()
  return loss.data.select(0, 0)

def train_decoder(encoder_cnn, decoder_rnn, loss_function, optimizer, image_caption_set, image_data_set, max_grad, useCuda=True):
  decoder_rnn.train()
  decoder_rnn.zero_grad()
  decoder_rnn.hidden = decoder_rnn.init_hidden()

  input_captions = data_loader.create_input_batch_captions([image_caption[1] for image_caption in image_caption_set], useCuda)
  target_captions = data_loader.create_target_batch_captions([image_caption[1] for image_caption in image_caption_set], useCuda)
  images = data_loader.create_input_batch_images(image_data_set, [image_caption[0] for image_caption in image_caption_set], useCuda)

  image_features = autograd.Variable(encoder_cnn(images).data)
  decoder_rnn(image_features)
  caption_scores, _ = decoder_rnn(input_captions)
  loss = loss_function(caption_scores, target_captions)
  loss.backward()
  nn.utils.clip_grad_norm(decoder_rnn.parameters(), max_grad)
  optimizer.step()
  return loss.data.select(0, 0)

def eval_decoder(encoder_cnn, decoder_rnn, loss_function, image_caption_set, image_data_set, useCuda=True):
  encoder_cnn.eval()
  decoder_rnn.eval()
  decoder_rnn.hidden = decoder_rnn.init_hidden()

  input_captions = data_loader.create_input_batch_captions([image_caption[1] for image_caption in image_caption_set], useCuda)
  target_captions = data_loader.create_target_batch_captions([image_caption[1] for image_caption in image_caption_set], useCuda)
  images = data_loader.create_input_batch_images(image_data_set, [image_caption[0] for image_caption in image_caption_set], useCuda)

  image_features = autograd.Variable(encoder_cnn(images).data)
  decoder_rnn(image_features)
  caption_scores, _ = decoder_rnn(input_captions)
  return loss_function(caption_scores, target_captions).data.select(0, 0)

def eval_decoder_random(encoder_cnn, decoder_rnn, loss_function, image_set, batched_data_set, word_to_index, batch_size=1, useCuda=True):
  decoder_rnn.eval()
  decoder_rnn.hidden = decoder_rnn.init_hidden()

  images, captions = data_loader.create_data_batch(image_set, batched_data_set, word_to_index, batch_size, useCuda)
  input_captions = data_loader.create_input_batch_captions(captions, useCuda)
  target_captions = data_loader.create_target_batch_captions(captions, useCuda)

  image_features = autograd.Variable(encoder_cnn(images).data)
  decoder_rnn(image_features)
  caption_scores, _ = decoder_rnn(input_captions)
  return loss_function(caption_scores, target_captions).data.select(0, 0)
