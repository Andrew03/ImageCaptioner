import os.path
import sys

def parse_train_params(file_name):
  if os.path.isfile(file_name) == True:
    params = open(file_name, 'r').read().splitlines()
    min_occurrences = int(params[0].split(" ")[1])
    batch_size = int(params[1].split(" ")[1])
    embedding_dim = int(params[2].split(" ")[1])
    hidden_size = int(params[3].split(" ")[1])
    dropout = float(params[4].split(" ")[1])
    decoder_lr = float(params[5].split(" ")[1])
    encoder_lr = float(params[6].split(" ")[1])
    num_epochs = int(params[7].split(" ")[1])
    grad_clip = int(params[8].split(" ")[1])
    isNormalized = params[9].split(" ")[1] == "True"
    useCuda = params[10].split(" ")[1] == "True"
    decayed_lr_rate = params[10].split(" ")[1]
    decayed_lr_step = params[10].split(" ")[1]
    return min_occurrences, batch_size, embedding_dim, hidden_size, dropout, \
      decoder_lr, encoder_lr, num_epochs, grad_clip, isNormalized, useCuda
  else:
    print("file name is invalid")
    return None 

def parse_run_params(file_name):
  if os.path.isfile(file_name) == True:
    params = open(file_name, 'r').read().splitlines()
    min_occurrences = int(params[0].split(" ")[1])
    batch_size = int(params[1].split(" ")[1])
    embedding_dim = int(params[2].split(" ")[1])
    hidden_size = int(params[3].split(" ")[1])
    dropout = float(params[4].split(" ")[1])
    decoder_lr = float(params[5].split(" ")[1])
    encoder_lr = float(params[6].split(" ")[1])
    num_epochs = int(params[7].split(" ")[1])
    grad_clip = int(params[8].split(" ")[1])
    num_runs = int(params[9].split(" ")[1])
    beam_size = int(params[10].split(" ")[1])
    printStepProb = params[11].split(" ")[1] == "True"
    isNormalized = params[12].split(" ")[1] == "True"
    useCuda = params[13].split(" ")[1] == "True"
    return min_occurrences, batch_size, embedding_dim, hidden_size, dropout, decoder_lr, \
      encoder_lr, num_epochs, grad_clip, num_runs, beam_size, printStepProb, isNormalized, useCuda
  else:
    print("file name is invalid")
    return None 
