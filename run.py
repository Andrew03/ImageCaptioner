import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as autograd
from pycocotools.coco import COCO
import pickle
import argparse
import sys
import smtplib
import getpass
import model
import trainer
import evaluator
import file_namer
from tqdm import tqdm
from build_vocab import Vocabulary
from batch_data import BatchedData
from batched_data_loader import get_loader
from plot import plot
from send_email import send_email

def main(args):
  # defining image size
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # PyTorch says images must be normalized like this
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])
  useCuda = not args.disable_cuda

  with open(args.vocab_path, 'rb') as f1, open(args.batched_val_path, 'rb') as f3:
    vocab = pickle.load(f1)
    batched_val_set = pickle.load(f3)

  batched_val_loader = get_loader(args.val_image_dir, args.val_caption_path, batched_val_set, vocab, transform, shuffle=True, num_workers=2)

  encoder_cnn = model.EncoderCNN(args.is_normalized, useCuda=useCuda)
  decoder_rnn = model.DecoderRNN(args.embedding_dim, args.hidden_size, len(vocab), 1, useCuda=useCuda)
  if torch.cuda.is_available() and useCuda:
    decoder_rnn.cuda()
  loss_function = nn.NLLLoss()

  checkpoint_name = args.load_checkpoint
  checkpoint = torch.load(checkpoint_name) if useCuda else torch.load(checkpoint_name, map_location=lambda storage, loc: storage)
  decoder_rnn.load_state_dict(checkpoint['state_dict'])
  args.load_checkpoint = checkpoint_name

  coco_caps=COCO(args.val_caption_path)
  for i, (image, _, img_id) in enumerate(batched_val_loader):
    if i == args.num_runs:
      break
    # beam search
    print("actual captions are:")
    annIds = coco_caps.getAnnIds(imgIds=img_id)
    anns = coco_caps.loadAnns(annIds)
    for ann in anns:
      print(ann['caption'])
    prediction = evaluator.beam_search(encoder_cnn, decoder_rnn, to_var(image, useCuda, volatile=True), vocab, args.beam_size, useCuda, args.print_step_prob)
    for caption in prediction:
      print("score is: " + str(caption[0]) + ", caption is: " + caption_to_string(caption[1], vocab))

def caption_to_string(caption, vocab):
  output = ""
  for value in caption:
    output += vocab(value) + " "
  return output

def to_var(x, useCuda=True, volatile=False):
  if torch.cuda.is_available() and useCuda:
    x = x.cuda()
  return autograd.Variable(x, volatile=volatile)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_runs', type=int,
                      default=10,
                      help='Number of images to generate captions for. Default value of 10')
  parser.add_argument('-use_train', action='store_true',
                      default=False,
                      help='Set to use training set instead of validation set')
  parser.add_argument('--vocab_path', type=str,
                      default='',
                      help='Path to vocab. Defaults to value generated by file_namer')
  parser.add_argument('--min_occurrences', type=int,
                      default=5,
                      help='Minimum occurrences of a word in the annotations. Default value of 5')
  parser.add_argument('--num_epochs', type=int,
                      default=10,
                      help='Number of epochs to train for. Default value of 10')
  parser.add_argument('--embedding_dim', type=int,
                      default=512,
                      help='Size of the embedding layer. Default value of 512')
  parser.add_argument('--hidden_size', type=int,
                      default=512,
                      help='Size of the hidden state. Default value of 512')
  parser.add_argument('-is_normalized', action='store_true',
                      default=False,
                      help='Set if encoder and decoder are normalized')
  parser.add_argument('-disable_cuda', action='store_true',
                      default=False,
                      help='Set if cuda should not be used')
  parser.add_argument('--load_checkpoint', type=str,
                      default='',
                      help='Saved checkpoint file name. Default behavior is to search using parameters')
  parser.add_argument('-print_step_prob', action='store_true',
                      default=False,
                      help='Set to show step probabilites')
  parser.add_argument('--beam_size', type=int,
                      default=5,
                      help='Size of beam. Default value of 5')
  args = parser.parse_args()

  # adding in arguments as needed
  if not args.vocab_path:
    args.vocab_path = file_namer.make_vocab_name_pkl(args.min_occurrences)
  if args.use_train:
    args.val_image_dir = 'data/train2014'
    args.val_caption_path = 'data/annotations/captions_train2014.json'
  else:
    args.val_image_dir = 'data/val2014'
    args.val_caption_path = 'data/annotations/captions_val2014.json'
  args.batched_val_path = file_namer.make_batch_name_pkl(1, isTrain=args.use_train)
  main(args)
