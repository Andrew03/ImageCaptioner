import torch
import os.path
import sys
sys.path.append(".")
print(os.getcwd())
import data_loader
import file_namer

min_occurrences = int(sys.argv[1])
batch_size = int(sys.argv[2])
vocab_file = file_namer.make_vocab_name(min_occurrences)
train_batch_file = file_namer.make_batch_name(batch_size, min_occurrences, isTrain=True)
val_batch_file = file_namer.make_batch_name(batch_size, min_occurrences, isTrain=False)

if not os.path.isfile(vocab_file) or not os.path.isfile(train_batch_file):
  train_set = data_loader.load_data(images='data/train2014', annotations='data/annotations/captions_train2014.json', transform=None)
  val_set = data_loader.load_data(images='data/val2014', annotations='data/annotations/captions_val2014.json', transform=None)
  word_to_index, index_to_word = data_loader.load_vocab(vocab_file)
  if word_to_index is None:
    word_to_index, index_to_word = data_loader.create_vocab(train_set, min_occurrence=min_occurrences)
    data_loader.write_vocab_to_file(index_to_word, vocab_file)
  print("batching data")
  batched_train_set = data_loader.batch_data(train_set, word_to_index, batch_size=batch_size)
  batched_val_set = data_loader.batch_data(val_set, word_to_index, batch_size=batch_size)
  data_loader.write_batched_data(batched_train_set, file_name=train_batch_file)
  data_loader.write_batched_data(batched_val_set, file_name=val_batch_file)
  print("built vocab and batched data")
else:
  print("vocab and batched data already exist")

