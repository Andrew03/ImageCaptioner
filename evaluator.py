import torch
import torch.autograd as autograd
import heapq
import operator
import random
from random import randint
import data_loader

def create_predict_batch(training_data, batched_data, useCuda=True):
  data_set = batched_data[random.choice(batched_data.keys())]
  image_caption = data_set[randint(0, len(data_set) - 1)]
  image, caption = training_data[image_caption[0]]
  images = data_loader.image_to_variable(torch.stack([image], 0), useCuda)
  return images, image_caption[0], caption

def create_predict_input_captions(captions, useCuda=True):
  return autograd.Variable(torch.cuda.LongTensor(captions)) if torch.cuda.is_available() and useCuda else autograd.Variable(torch.LongTensor(captions))

def beam_search(encoder_cnn, decoder_rnn, images, beam_size=1, useCuda=True, printStepProbs=False):
  decoder_rnn.eval()
  #best_phrases = [[0, []] for i in range(beam_size)]
  completed_phrases = []
  index = 0
  # intializing hidden state with image
  decoder_rnn.hidden = decoder_rnn.init_hidden(1)
  decoder_rnn(encoder_cnn(images))
  # creating intial input batch
  input_captions = create_predict_input_captions([1], useCuda)
  initial_scores, initial_probabilities = decoder_rnn(input_captions)
  # getting top scores
  top_scores, top_indices = initial_scores.topk(beam_size)
  step_score = 0
  top_probs, _ = initial_probabilities.topk(beam_size)
  for score in top_probs[0].data:
    step_score += score
  if printStepProbs:
    print(str(index) + ": " + str(step_score))
  # updating best phrases
  best_phrases = [[top_scores[0].data[i], [top_indices[0].data[i]]] for i in range(beam_size)]
  # getting next batch of inputs
  next_captions = top_indices.resize(beam_size, 1)
  decoder_rnn.hidden = (decoder_rnn.hidden[0].repeat(1, beam_size, 1), decoder_rnn.hidden[1].repeat(1, beam_size, 1))
  while index < 20:
    index += 1
    scores, probabilities = decoder_rnn(next_captions)
    best_candidates = []
    top_scores, top_indices = scores.topk(beam_size)
    top_probs, _ = probabilities.topk(beam_size)
    step_scores = []
    for i in range(len(best_phrases)):
      step_score = 0
      for score in top_probs[i].data:
        step_score += score
      step_scores.append(step_score)
    if printStepProbs:
      print(str(index) + ": " + str(step_scores))
    len_phrases = len(best_phrases[0][1])
    for i in range(len(best_phrases)):
      for j in range(beam_size):
        best_candidates.extend([[(best_phrases[i][0] * len_phrases + top_scores[i].data[j]) / (len_phrases + 1),
          best_phrases[i][1] + [top_indices[i].data[j]],
          i]])
    best_candidates = sorted(best_candidates, key=lambda score_caption: score_caption[0])[-beam_size:]
    for phrase in best_candidates:
      if phrase[1][-1] == 2:
        best_candidates.remove(phrase)
        completed_phrases.append([phrase[0], phrase[1]])
    if len(completed_phrases) >= beam_size:
      return completed_phrases
    best_phrases = [[phrase[0], phrase[1]] for phrase in best_candidates]
    next_captions = [[phrase[1][-1]] for phrase in best_phrases]
    next_captions = create_predict_input_captions(next_captions, useCuda)
    decoder_rnn.hidden = (torch.stack([decoder_rnn.hidden[0][0].select(0, phrase[2]) for phrase in best_candidates]),
      torch.stack([decoder_rnn.hidden[1][0].select(0, phrase[2]) for phrase in best_candidates]))
    decoder_rnn.hidden = (decoder_rnn.hidden[0].unsqueeze(0), decoder_rnn.hidden[1].unsqueeze(0))
  return completed_phrases + best_phrases
