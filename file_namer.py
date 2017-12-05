import os.path

def make_vocab_name(min_occurrences):
  return "data/vocab/vocab_occurrence_" + str(min_occurrences) + ".txt"

def make_batch_name(batch_size, min_occurrences, isTrain):
  return "data/batched_data/" + ("train" if isTrain else "val") + \
    "_batch_" + str(batch_size) + "_occurrence_" + str(min_occurrences) + ".txt"

def make_output_name(batch_size, min_occurrences, num_epochs, dropout, model_lr, \
  encoder_lr, embedding_dim, hidden_size, grad_clip, isTrain, isNormalized):
  return "output/" + ("train" if isTrain else "val") + \
    "_batch_" + str(batch_size) + "_occurrence_" + str(min_occurrences) + \
    "_epoch_" + str(num_epochs) + "_dropout_" + str(dropout) + "_modelLR_" + \
    str(model_lr) + "_encoderLR_" + str(encoder_lr) + "_dim_" + str(embedding_dim) \
    + "x" +str(hidden_size) + "_clip_" + str(grad_clip) + \
    ("isnorm" if isNormalized else "nonorm") + ".txt"

def make_save_name(batch_size, min_occurrences, num_epochs, dropout, model_lr, \
  encoder_lr, embedding_dim, hidden_size, grad_clip, isNormalized):
  return "model/model_batch_" + str(batch_size) + "_occurrence_" + \
  str(min_occurrences) + "_epoch_" + str(num_epochs) + "_dropout_" + str(dropout) + \
  "_modelLR_" + str(model_lr) + "_encoderLR_" + str(encoder_lr) + "_dim_" + \
  str(embedding_dim) + "x" + str(hidden_size) + "_clip_" + str(grad_clip) + \
  ("isnorm" if isNormalized else "nonorm") + ".pt"

def get_checkpoint(model_name, epoch_prefix="_epoch_", separator="_"):
  epoch_start_index = model_name.find(epoch_prefix) + len(epoch_prefix)
  epoch_end_index = model_name.find(separator, epoch_start_index)
  num_epochs = int(model_name[epoch_start_index : epoch_end_index])
  for i in range(num_epochs, 0, -1):
    checkpoint_name = model_name[:epoch_start_index] + str(i) + \
      model_name[epoch_end_index:]
    if os.path.isfile(checkpoint_name):
      return checkpoint_name
  return None
