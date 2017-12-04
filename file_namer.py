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
