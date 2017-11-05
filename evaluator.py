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

def create_predict_input_captions(captions, batch_size, beam_size=1):
  #print(captions)
  while len(captions) < batch_size:
    captions.append([0 for i in range(len(captions[0]))])
  return autograd.Variable(torch.cuda.LongTensor(captions)) if torch.cuda.is_available() else autograd.Variable(torch.LongTensor(captions))
  
'''
  image is a batch where 
'''
def beam_search(cnn, lstm, images, batch_size, beam_size=1):
  model = lstm.eval()
  best_phrases = [[0, [1]]]
  completed_phrases = []
  image_features = cnn(images)
  index = 0
  while index < 20 and len(completed_phrases) < beam_size:
    model.hidden = model.init_hidden()
    model(image_features)
    input_captions = create_predict_input_captions([score_caption[1] for score_caption in best_phrases], batch_size, beam_size=beam_size)
    init_score = model(input_captions)
    #scores = model(input_captions).data[-batch_size:]
    scores= init_score.data[-batch_size:]
    best_candidates = []
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
