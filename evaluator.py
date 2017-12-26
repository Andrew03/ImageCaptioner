import torch
import torch.autograd as autograd
import heapq
import operator
import random
from random import randint
from torch.nn.utils.rnn import pack_padded_sequence
import data_loader

def to_var(x, useCuda=True, volatile=False):
  if torch.cuda.is_available() and useCuda:
    x = x.cuda()
  return autograd.Variable(x, volatile=volatile)

def evaluate(encoder_cnn, decoder_rnn, loss_function, images, captions, lengths, useCuda):
  encoder_cnn.eval()
  decoder_rnn.eval()
  decoder_rnn.hidden = decoder_rnn.init_hidden()

  input_images = to_var(images, useCuda, volatile=True)
  # stripping away the <EOS> token from inputs
  input_captions = to_var(torch.index_select(captions, 1, torch.LongTensor([i for i in range(lengths[0] - 1)])), useCuda, volatile=True)
  # stripping away the <SOS> token from targets
  targets = to_var(torch.index_select(captions, 1, torch.LongTensor([i for i in range(1, lengths[0])])), useCuda, volatile=True)
  target_captions = pack_padded_sequence(targets, lengths, batch_first=True)[0]

  decoder_rnn(autograd.Variable(encoder_cnn(input_images).data))
  caption_scores, _ = decoder_rnn(input_captions)
  return loss_function(caption_scores, target_captions).data.select(0, 0)

def create_predict_batch(training_data, batched_data, useCuda=True):
  data_set = batched_data[random.choice(batched_data.keys())]
  image_caption = data_set[randint(0, len(data_set) - 1)]
  image, caption = training_data[image_caption[0]]
  images = data_loader.image_to_variable(torch.stack([image], 0), useCuda)
  return images, image_caption[0], caption

def create_predict_input_captions(captions, useCuda=True):
  return autograd.Variable(torch.cuda.LongTensor(captions)) if torch.cuda.is_available() and useCuda else autograd.Variable(torch.LongTensor(captions))

def beam_search(encoder_cnn, decoder_rnn, image, beam_size=1, useCuda=True, printStepProbs=False):
  decoder_rnn.eval()
  completed_phrases = []
  index = 0
  # intializing hidden state with image
  decoder_rnn.hidden = decoder_rnn.init_hidden(1)
  decoder_rnn(encoder_cnn(image))
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
    top_candidates = sorted(best_candidates, key=lambda score_caption: score_caption[0])[-beam_size:]
    for phrase in top_candidates:
      if phrase[1][-1] == 2:
        top_candidates.remove(phrase)
        completed_phrases.append([phrase[0], phrase[1]])
    if len(completed_phrases) >= beam_size:
      return completed_phrases
    best_phrases = [[phrase[0], phrase[1]] for phrase in top_candidates]
    next_captions = create_predict_input_captions([[phrase[1][-1]] for phrase in best_phrases], useCuda)
    decoder_rnn.hidden = (torch.stack([decoder_rnn.hidden[0][0].select(0, phrase[2]) for phrase in top_candidates]).unsqueeze(0),
      torch.stack([decoder_rnn.hidden[1][0].select(0, phrase[2]) for phrase in top_candidates]).unsqueeze(0))
  return completed_phrases + best_phrases
