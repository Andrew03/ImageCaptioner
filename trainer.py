import torch.autograd as autograd
import torch.nn as nn
import data_loader

def train_model(cnn, lstm, loss_function, optimizer, image_caption_set, image_data_set, max_grad):
  input_captions = data_loader.create_input_batch_captions([image_caption[1] for image_caption in image_caption_set])
  target_captions = data_loader.create_target_batch_captions([image_caption[1] for image_caption in image_caption_set])
  images = data_loader.create_input_batch_images(image_data_set, [image_caption[0] for image_caption in image_caption_set])
  image_features = autograd.Variable(cnn(images).data)
  lstm.train()
  lstm.zero_grad()
  lstm.hidden = lstm.init_hidden()
  lstm(image_features)
  caption_scores = lstm(input_captions)
  loss = loss_function(caption_scores, target_captions)
  loss.backward()
  nn.utils.clip_grad_norm(lstm.parameters(), max_grad)
  optimizer.step()
  return loss.data.select(0, 0)

def eval_model(cnn, lstm, loss_function, image_caption_set, image_data_set):
  input_captions = data_loader.create_input_batch_captions([image_caption[1] for image_caption in image_caption_set])
  target_captions = data_loader.create_target_batch_captions([image_caption[1] for image_caption in image_caption_set])
  images = data_loader.create_input_batch_images(image_data_set, [image_caption[0] for image_caption in image_caption_set])
  image_features = autograd.Variable(cnn(images).data)
  lstm.eval()
  lstm.hidden = lstm.init_hidden()
  lstm(image_features)
  caption_scores = lstm(input_captions)
  return loss_function(caption_scores, target_captions).data.select(0, 0)

def eval_model_random(cnn, lstm, loss_function, image_set, batched_data_set, word_to_index, batch_size=1):
  images, captions = data_loader.create_data_batch(image_set, batched_data_set, word_to_index, batch_size=batch_size)
  input_captions = data_loader.create_input_batch_captions(captions)
  target_captions = data_loader.create_target_batch_captions(captions)
  image_features = autograd.Variable(cnn(images).data)
  lstm.eval()
  lstm.hidden = lstm.init_hidden()
  lstm(image_features)
  caption_scores = lstm(input_captions)
  return loss_function(caption_scores, target_captions).data.select(0, 0)
