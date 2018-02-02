import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
from file_namer import make_vocab_name_pkl

class Vocabulary(object):
  """A vocabulary wrapper, contains a word_to_index dictionary and a index_to_word list"""
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = []
    self.index = 0

  def add_word(self, word):
    if not word in self.word_to_index:
      self.word_to_index[word] = self.index
      self.index_to_word.append(word)
      self.index += 1

  def __call__(self, word):
    if type(word) == str:
      if not word in self.word_to_index:
        return self.word_to_index['<UNK>']
      return self.word_to_index[word]
    else:
      return self.index_to_word[word]

  def __len__(self):
    return self.index

def build_vocab(caption_path, min_occurrences):
  """Builds a Vocabulary object"""
  coco = COCO(caption_path)
  counter = Counter()
  ids = coco.anns.keys()
  for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)

    if i % 1000 == 0:
      print("Tokenized [%d/%d] captions." %(i, len(ids)))

  # a word must appear at least min_occurrence times to be included in the vocabulary
  words = [word for word, count in counter.items() if count >= min_occurrences]

  # Creating a vocabulary object
  vocab = Vocabulary()
  vocab.add_word('<SOS>')
  vocab.add_word('<EOS>')
  vocab.add_word('<UNK>')

  # Adds the words from the captions to the vocabulary
  for word in words:
    vocab.add_word(word)
  return vocab

def main(args):
  vocab = build_vocab(caption_path=args.caption_path,
                      min_occurrences=args.min_occurrences)
  save_path = args.save_path if args.save_path != "" else make_vocab_name_pkl(args.min_occurrences)
  with open(save_path, 'wb') as f:
    pickle.dump(vocab, f)
  print("Total vocabulary size: %d" %len(vocab))
  print("Saved the vocabulary at '%s'" %save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--caption_path', type=str, 
                      default='./data/annotations/captions_train2014.json', 
                      help='Path for train annotation file. Default value of ./data/annotations/captions_train2014.json')
  parser.add_argument('--save_path', type=str,
                      default='', 
                      help='Path to save vocabulary. Defaults to value generated by file_namer')
  parser.add_argument('--min_occurrences', type=int,
                      default=5, 
                      help='Minimum occurrences of a word in the annotations. Default value of 5')
  args = parser.parse_args()
  main(args)
