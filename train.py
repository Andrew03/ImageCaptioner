import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as autograd
import pickle
import argparse
import os
import smtplib
import getpass
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')
import model
import trainer
import evaluator
import file_namer
from tqdm import tqdm
tqdm.monitor_interval = 0
from build_vocab import Vocabulary
from batch_data import BatchedData
from batched_data_loader import get_loader
from plot import plot
from send_email import send_email

def validate(random_val_loader, encoder_cnn, decoder_rnn, loss_function, useCuda, iter_number, return_index, return_value):
  decoder_rnn = decoder_rnn.copy()
  val_sum_loss = 0
  for i, (val_images, val_captions, _) in enumerate(random_val_loader, 1):
    val_sum_loss += evaluator.evaluate(encoder_cnn, decoder_rnn, loss_function, val_images, val_captions, useCuda)
    if i == 100:
      break
  return_index.value = iter_number
  return_value.value = val_sum_loss / 100

def validate_full(batched_val_loader, encoder_cnn, decoder_rnn, loss_function, useCuda, epoch, num_epochs, len_batched_train_loader, return_index, return_value):
  decoder_rnn = decoder_rnn.copy()
  val_progress_bar = tqdm(iterable=batched_val_loader, desc='Epoch [%i/%i] (Val)' %(epoch, num_epochs), position=1)
  val_sum_loss = 0
  for i, (images, captions, _) in enumerate(val_progress_bar):
    val_sum_loss += evaluator.evaluate(encoder_cnn, decoder_rnn, loss_function, images, captions, useCuda)
    val_progress_bar.set_postfix(loss = val_sum_loss / i if i > 0 else 1) 
  return_index.value = (epoch + 1) * len_batched_train_loader
  return_value.value = val_sum_loss / len(batched_val_loader)

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

  with open(args.vocab_path, 'rb') as f1, open(args.batched_train_path, 'rb') as f2, open(args.batched_val_path, 'rb') as f3:
    vocab = pickle.load(f1)
    batched_train_set = pickle.load(f2)
    batched_val_set = pickle.load(f3)

  batched_train_loader = get_loader(args.train_image_dir, args.train_caption_path, batched_train_set, vocab, transform, shuffle=True, num_workers=3)
  batched_val_loader = get_loader(args.val_image_dir, args.val_caption_path, batched_val_set, vocab, transform, shuffle=True, num_workers=1)
  random_val_loader = get_loader(args.val_image_dir, args.val_caption_path, batched_val_set, vocab, transform, shuffle=True, num_workers=1)

  encoder_cnn = model.EncoderCNN(args.is_normalized, useCuda=useCuda)
  decoder_rnn = model.DecoderRNN(args.embedding_dim, args.hidden_size, len(vocab), args.batch_size, dropout=args.dropout, useCuda=useCuda)
  if torch.cuda.is_available() and useCuda:
    decoder_rnn.cuda()
  loss_function = nn.NLLLoss()
  optimizer = optim.Adam([
    {'params': decoder_rnn.word_embedding_layer.parameters()},
    {'params': decoder_rnn.lstm.parameters()},
    {'params': decoder_rnn.hidden2word.parameters()},
    {'params': decoder_rnn.image_embedding_layer.parameters(), 'lr': args.encoder_lr},
    ], lr=args.decoder_lr)

  output_train_file = open(args.output_train_name, 'w')
  output_val_file = open(args.output_val_name, 'w')
  start_epoch = 0

  save_name = file_namer.make_checkpoint_name(args.batch_size, args.min_occurrences, args.num_epochs, \
    args.dropout, args.decoder_lr, args.encoder_lr, args.embedding_dim, args.hidden_size, args.grad_clip, \
    args.is_normalized) if args.load_checkpoint == "" else args.load_checkpoint
  checkpoint_name = file_namer.get_checkpoint(save_name)
  if checkpoint_name is not None:
    print("loading from checkpoint " + checkpoint_name)
    checkpoint = torch.load(checkpoint_name) if useCuda else torch.load(checkpoint_name, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    decoder_rnn.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.load_checkpoint = checkpoint_name
    checkpoint = None 
    torch.cuda.empty_cache()
  else:
    print("No existing checkpoints, starting from scratch")
    args.load_checkpoint = "No checkpoint found"

  full_return_index = mp.Value('i', 0)
  full_return_value = mp.Value('d', 0.0)
  full_val_processes = None
  for epoch in range(start_epoch, args.num_epochs):
    val_processes = None
    return_index = mp.Value('i', 0)
    return_value = mp.Value('d', 0.0)
    train_progress_bar = tqdm(iterable=batched_train_loader, desc='Epoch [%i/%i] (Train)' %(epoch, args.num_epochs))
    train_sum_loss = 0
    for i, (images, captions, _) in enumerate(train_progress_bar):
      train_sum_loss += trainer.train(encoder_cnn, decoder_rnn, loss_function, optimizer, images, captions, args.grad_clip, useCuda)
      train_progress_bar.set_postfix(loss = train_sum_loss / ((i % 100) + 1))
      if i % 100 == 0:
        output_train_file.write("%d, %5.4f\n" %(epoch * len(batched_train_loader) + i, train_sum_loss / 100 if i > 0 else train_sum_loss))
        if i % 1000 == 0:
          if val_processes is not None:
            val_processes.join()
            output_val_file.write("%d, %5.4f\n" %(return_index.value, return_value.value))
          val_processes = mp.Process(target=validate, args=(random_val_loader, encoder_cnn, decoder_rnn, loss_function, useCuda, epoch * len(batched_train_loader) + i, return_index, return_value))
          val_processes.start()
        train_sum_loss = 0

    if full_val_processes is not None:
      full_val_processes.join()
      output_val_file.write("End of Epoch\n%d, %5.4f\n" %(full_return_index.value, full_return_value.value))
    full_val_processes = mp.Process(target=validate_full, args=(batched_val_loader, encoder_cnn, decoder_rnn, loss_function, useCuda, epoch, args.num_epochs, len(batched_train_loader), full_return_index, full_return_value))
    full_val_processes.start()
    torch.save({'epoch': epoch + 1,
                'state_dict': decoder_rnn.state_dict(),
                'optimizer': optimizer.state_dict()},
                file_namer.make_checkpoint_name(args.batch_size, args.min_occurrences, epoch + 1, args.dropout, \
                args.decoder_lr, args.encoder_lr, args.embedding_dim, args.hidden_size, args.grad_clip, args.is_normalized))
  if full_val_processes:
    full_val_processes[-1].join()
    output_val_file.write("End of Epoch\n%d, %5.4f\n" %(full_return_index.value, full_return_value.value))
    del full_val_processes[-1]

  output_train_file.close()
  output_val_file.close()

  if args.plot:
    args.train_files.append(args.output_train_name)
    args.val_files.append(args.output_val_name)
    plot(args)
    args.png_files = [args.plot_name]
  if args.send_email:
    args.txt_files = [args.output_train_name, args.output_val_name]
    f = open('arguments.txt', 'w')
    for arg in sorted(vars(args)):
      # arguments we don't want sent in the email
      ignore_args = ['user', 'password', 'to', 'plot_name', 'train_image_dir', 'val_image_dir',
        'send_email', 'plot', 'plot_name', 'train_caption_path', 'val_caption_path', 'png_files',
        'txt_files', 'disable_cuda', 'body', 'output_train_name', 'output_val_name', 'show', 'subject', 'max_batched_set_size']
      if not arg in ignore_args:
        f.write("%s: %s\n" %(arg, getattr(args, arg)))
    f.close()
    if not args.body:
      args.body = 'arguments.txt'
    else:
      args.txt_files.append('arguments.txt')
    send_email(args)

if __name__ == '__main__':
  #mp.set_start_method("spawn") 
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_caption_path', type=str,
                      default='data/annotations/captions_train2014.json',
                      help='Path for train annotation file. Default value of ./data/annotations/captions_train2014.json')
  parser.add_argument('--train_image_dir', type=str,
                      default='data/train2014',
                      help='Path to train image directory. Default value of ./data/train2014')
  parser.add_argument('--batched_train_path', type=str,
                      default='',
                      help='Path to batched train set. Defaults to value generated by file_namer')
  parser.add_argument('--val_caption_path', type=str,
                      default='data/annotations/captions_val2014.json',
                      help='Path for validation annotation file. Default value of ./data/annotations/captions_val2014.json')
  parser.add_argument('--val_image_dir', type=str,
                      default='data/val2014',
                      help='Path to val image directory. Default value of ./data/val2014')
  parser.add_argument('--batched_val_path', type=str,
                      default='',
                      help='Path to batched val set. Defaults to value generated by file_namer')
  parser.add_argument('--max_batched_set_size', type=int,
                      default=None,
                      help='Maximum size of the batched data set. Defaults to no maximum')
  parser.add_argument('--vocab_path', type=str,
                      default='',
                      help='Path to vocab. Defaults to value generated by file_namer')
  parser.add_argument('--output_train_name', type=str,
                      default='',
                      help='Output train file name. Defaults to value generated by file_namer')
  parser.add_argument('--output_val_name', type=str,
                      default='',
                      help='Output val file name. Defaults to value generated by file_namer')
  parser.add_argument('--min_occurrences', type=int,
                      default=5,
                      help='Minimum occurrences of a word in the annotations. Default value of 5')
  parser.add_argument('--batch_size', type=int,
                      default=32,
                      help='Size of a batch. Default value of 32')
  parser.add_argument('--num_epochs', type=int,
                      default=10,
                      help='Number of epochs to train for. Default value of 10')
  parser.add_argument('--embedding_dim', type=int,
                      default=512,
                      help='Size of the embedding layer. Default value of 512')
  parser.add_argument('--hidden_size', type=int,
                      default=512,
                      help='Size of the hidden state. Default value of 512')
  parser.add_argument('--encoder_lr', type=float,
                      default=0.001,
                      help='Learning rate for feature mapping layer. Default value of 0.001')
  parser.add_argument('--decoder_lr', type=float,
                      default=0.001,
                      help='Learning rate for decoder. Default value of 0.001')
  parser.add_argument('--grad_clip', type=float,
                      default=5,
                      help='Maximum gradient. Default value of 5')
  parser.add_argument('--dropout', type=float,
                      default=0.0,
                      help='Dropout value for the decoder. Default value of 0.0')
  parser.add_argument('-is_normalized', action='store_true',
                      default=False,
                      help='Set if encoder and decoder are normalized')
  parser.add_argument('-disable_cuda', action='store_true',
                      default=False,
                      help='Set if cuda should not be used')
  parser.add_argument('--load_checkpoint', type=str,
                      default='',
                      help='Saved checkpoint file name. Default behavior is to search using parameters')
  parser.add_argument('-send_email', action='store_true',
                      default=False,
                      help='Set if email should be sent on completion')
  parser.add_argument('--user', type=str,
                      default='',
                      help='Email address of the user')
  parser.add_argument('--to', type=str,
                      default='',
                      help='Email address of the receiver')
  parser.add_argument('--subject', type=str,
                      default='Finished Training',
                      help='Subject of the email. Default value of \'Finished Training\'')
  parser.add_argument('--body', type=str,
                      default='',
                      help='File containing body of the message to send to the receiver')
  parser.add_argument('-plot', action='store_true',
                      default=False,
                      help='Set if loss should be plotted')
  parser.add_argument('--plot_name', type=str,
                      default='graph.png',
                      help='Path to save the plot to. Default value of graph.png')
  parser.add_argument('--prev_train_file', action='append', dest='train_files',
                      default=[],
                      help='File containing training iteration and loss on each line, separated by a comma')
  parser.add_argument('--prev_val_file', action='append', dest='val_files',
                      default=[],
                      help='File containing validation iteration and loss on each line, separated by a comma')
  args = parser.parse_args()

  # adding in arguments as needed
  args.show = False
  if not args.vocab_path :
    args.vocab_path = file_namer.make_vocab_name_pkl(args.min_occurrences)
  if not args.batched_train_path:
    args.batched_train_path = file_namer.make_batch_name_pkl(args.batch_size, args.max_batched_set_size, True)
  if not args.batched_val_path:
    args.batched_val_path = file_namer.make_batch_name_pkl(args.batch_size, args.max_batched_set_size, False)
  if not args.output_train_name:
    args.output_train_name = file_namer.make_output_name(args.batch_size, \
      args.min_occurrences, args.num_epochs, args.dropout, args.decoder_lr, args.encoder_lr, \
      args.embedding_dim, args.hidden_size, args.grad_clip, args.is_normalized, True)
  if not args.output_val_name:
    args.output_val_name = file_namer.make_output_name(args.batch_size, \
      args.min_occurrences, args.num_epochs, args.dropout, args.decoder_lr, args.encoder_lr, \
      args.embedding_dim, args.hidden_size, args.grad_clip, args.is_normalized, False)
  args.png_files = []
  if args.send_email:
    args.password = getpass.getpass('Password: ')

    # checking to make sure email credentials are valid
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(args.user, args.password)
    server.close()
  #try:
  main(args)
  """
  except:
    if args.send_email:
      f = open('arguments.txt', 'w')
      for arg in sorted(vars(args)):
        # arguments we don't want sent in the email
        ignore_args = ['user', 'password', 'to', 'plot_name', 'train_image_dir', 'val_image_dir',
          'send_email', 'plot', 'plot_name', 'train_caption_path', 'val_caption_path', 'png_files',
          'txt_files', 'disable_cuda', 'body', 'output_train_name', 'output_val_name', 'show', 'subject', 'max_batched_set_size']
        if not arg in ignore_args:
          f.write("%s: %s\n" %(arg, getattr(args, arg)))
      f.close()
      if not args.body:
        args.body = 'arguments.txt'
      else:
        args.txt_files.append('arguments.txt')
      if args.subject == 'Finished Training':
        args.subject = 'Run failed'
      else:
        args.subject += ': Run failed'
      args.txt_files = []
      args.png_files = []
      send_email(args)
    """
