import torch
import torch.autograd as autograd
import heapq
import operator
import random
from random import randint
import data_loader

def create_predict_batch(training_data, batched_data, batch_size=1):
  data_set = batched_data[random.choice(batched_data.keys())]
  image_caption = data_set[randint(0, len(data_set) - 1)]
  image, caption = training_data[image_caption[0]]
  images = data_loader.image_to_variable(torch.stack([image], 0))
  return images, image_caption[0], caption

def create_predict_input_captions(captions):
  return autograd.Variable(torch.cuda.LongTensor(captions)) if torch.cuda.is_available() else autograd.Variable(torch.LongTensor(captions))

def beam_search(cnn, lstm, images, beam_size=1):
  lstm.eval()
  best_phrases = [[0, []] for i in range(beam_size)]
  completed_phrases = []
  index = 0
  # intializing hidden state with image
  lstm.hidden = lstm.init_hidden(1)
  lstm(cnn(images))
  # creating intial input batch
  input_captions = create_predict_input_captions([1])
  initial_score = lstm(input_captions).data[0]
  # getting top scores
  top_indices = zip(*heapq.nlargest(beam_size, enumerate(initial_score), key=operator.itemgetter(1)))[0]
  # updating best phrases
  best_phrases = [[best_phrases[0][0] + initial_score[score_index], best_phrases[0][1] + [score_index]] for score_index in top_indices]
  # getting next batch of inputs
  next_captions = create_predict_input_captions(list(top_indices))
  lstm.hidden = (lstm.hidden[0].repeat(1, beam_size, 1), lstm.hidden[1].repeat(1, beam_size, 1))
  while index < 20:
    index += 1
    scores = lstm(next_captions).data
    best_candidates = []
    for i in range(len(best_phrases)):
      score = scores[i]
      top_indices = zip(*heapq.nlargest(beam_size, enumerate(score), key=operator.itemgetter(1)))[0]
      best_candidates.extend([[(best_phrases[i][0] * len(best_phrases[i][1]) + score[score_index]) / (len(best_phrases[i][1]) + 1),
        best_phrases[i][1] + [score_index],
        i] for score_index in top_indices])
    best_candidates = sorted(best_candidates, key=lambda score_caption: score_caption[0])[-beam_size:]
    for phrase in best_candidates:
      if phrase[1][-1] == 2:
        best_candidates.remove(phrase)
        completed_phrases.append([phrase[0], phrase[1]])
    if len(completed_phrases) >= beam_size:
      return completed_phrases
    best_phrases = [[phrase[0], phrase[1]] for phrase in best_candidates]
    next_captions = [[phrase[1][-1]] for phrase in best_phrases]
    next_captions = create_predict_input_captions(next_captions)
    lstm.hidden = (torch.stack([lstm.hidden[0][0].select(0, phrase[2]) for phrase in best_candidates]),
      torch.stack([lstm.hidden[1][0].select(0, phrase[2]) for phrase in best_candidates]))
    lstm.hidden = (lstm.hidden[0].unsqueeze(0), lstm.hidden[1].unsqueeze(0))
  return completed_phrases + best_phrases
