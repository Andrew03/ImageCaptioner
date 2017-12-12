import torch.autograd as autograd
import torch.nn as nn
import data_loader

def train_model(cnn, lstm, loss_function, optimizer, image_caption_set, image_data_set, max_grad, useCuda=True):
  lstm.train()
  lstm.zero_grad()
  lstm.hidden = lstm.init_hidden()

  input_captions = data_loader.create_input_batch_captions([image_caption[1] for image_caption in image_caption_set], useCuda)
  target_captions = data_loader.create_target_batch_captions([image_caption[1] for image_caption in image_caption_set], useCuda)
  images = data_loader.create_input_batch_images(image_data_set, [image_caption[0] for image_caption in image_caption_set], useCuda)

  image_features = autograd.Variable(cnn(images).data)
  lstm(image_features)
  caption_scores, _ = lstm(input_captions)
  loss = loss_function(caption_scores, target_captions)
  loss.backward()
  nn.utils.clip_grad_norm(lstm.parameters(), max_grad)
  optimizer.step()
  return loss.data.select(0, 0)

def eval_model(cnn, lstm, loss_function, image_caption_set, image_data_set, useCuda=True):
  lstm.eval()
  lstm.hidden = lstm.init_hidden()

  input_captions = data_loader.create_input_batch_captions([image_caption[1] for image_caption in image_caption_set], useCuda)
  target_captions = data_loader.create_target_batch_captions([image_caption[1] for image_caption in image_caption_set], useCuda)
  images = data_loader.create_input_batch_images(image_data_set, [image_caption[0] for image_caption in image_caption_set], useCuda)

  image_features = autograd.Variable(cnn(images).data)
  lstm(image_features)
  caption_scores, _ = lstm(input_captions)
  return loss_function(caption_scores, target_captions).data.select(0, 0)

def eval_model_random(cnn, lstm, loss_function, image_set, batched_data_set, word_to_index, batch_size=1, useCuda=True):
  lstm.eval()
  lstm.hidden = lstm.init_hidden()

  images, captions = data_loader.create_data_batch(image_set, batched_data_set, word_to_index, batch_size, useCuda)
  input_captions = data_loader.create_input_batch_captions(captions, useCuda)
  target_captions = data_loader.create_target_batch_captions(captions, useCuda)

  image_features = autograd.Variable(cnn(images).data)
  lstm(image_features)
  caption_scores, _ = lstm(input_captions)
  return loss_function(caption_scores, target_captions).data.select(0, 0)
