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
  image, _ = training_data[image_caption[0]]
  images = [image for i in range(batch_size)]
  captions = [image_caption[1] for i in range(batch_size)] if batch_size > 1 else image_caption[1]
  images = data_loader.image_to_variable(torch.stack(images, 0))
  return images, captions

def create_predict_input_captions(captions):
  return autograd.Variable(torch.cuda.LongTensor(captions)) if torch.cuda.is_available() else autograd.Variable(torch.LongTensor(captions))
  
'''
  image is a batch where 
'''
def beam_search(cnn, lstm, images, beam_size=1):
  lstm.eval()
  best_phrases = [[0, [1]]]
  completed_phrases = []
  image_features = cnn(images)
  index = 0
  lstm.hidden = lstm.init_hidden(1)
  lstm(image_features)
  input_captions = create_predict_input_captions([score_caption[1]])
  init_score = lstm(input_captions)
  lstm.hidden = (lstm.hidden[0].repeat(1, beam_size, 1), lstm.hidden[1].repeat(1, beam_size, 1))
  while index < 20 and len(completed_phrases) < beam_size:
    input_captions = create_predict_input_captions([score_caption[1]])
    init_score = lstm(input_captions)
    scores= init_score.data[-beam_size:]
    best_candidates = []
    lstm.hidden = (lstm.hidden[0].repeat(1, beam_size, 1), lstm.hidden[1].repeat(1, beam_size, 1))
    lstm.hidden = lstm.hidden[0][0].select(0, index)
    # changed up to here
    for i in range(len(best_phrases)):
      score = scores[i]
      top_indices = zip(*heapq.nlargest(beam_size, enumerate(score), key=operator.itemgetter(1)))[0]
      #print(top_indices)
      best_candidates.extend([[best_phrases[i][0] + score[score_index], best_phrases[i][1] + [score_index]] for score_index in top_indices])
    #print(best_candidates)
    best_phrases = sorted(best_candidates, key=lambda score_caption: score_caption[0])[-beam_size:]
    #best_phrases = sorted(best_candidates, key=lambda score_caption: score_caption[0])
    #print("best phrases are")
    #print(best_phrases)
    for score_caption in best_phrases:
      #print(score_caption)
      if score_caption[1][-1] == 2:
        completed_phrases.append(score_caption)
        best_phrases.remove(score_caption)
    index += 1
  if len(completed_phrases) < beam_size:
    completed_phrases.extend(best_phrases)
  #return sorted(completed_phrases, key=lambda score_caption: score_caption[0])[-1]
  return sorted(completed_phrases, key=lambda score_caption: score_caption[0])[-beam_size:]

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
