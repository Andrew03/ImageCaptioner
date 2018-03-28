import pylab as plt
import matplotlib.patches as mpatches
import csv
import sys
import argparse

def plot(args):
  train_loss = []
  train_trial = [] 
  val_loss = []
  val_trial = [] 
  epoch_loss = []
  epoch_trial = [] 
  if len(args.train_files) == 0 and len(args.val_files) == 0:
    print("Needs a file to plot with!")
    sys.exit()
  else:
    for file_name in args.train_files:
      with open(file_name, 'r') as file:
        for line in file:
          number_value = line.split(",")
          train_loss.append(float(number_value[1]))
          train_trial.append(int(number_value[0]))
    for file_name in args.val_files:
      with open(file_name, 'r') as file:
        endOfEpoch = False
        for line in file:
          number_value = line.split(",")
          if endOfEpoch == True:
            epoch_loss.append(float(number_value[1]))
            epoch_trial.append(int(number_value[0]))
            endOfEpoch = False
          else:
            if len(number_value) == 1:
              endOfEpoch = True
            else:
              val_loss.append(float(number_value[1]))
              val_trial.append(int(number_value[0]))
  # plotting the data
  plt.plot(train_trial, train_loss, 'go', markersize=1, label='Training Batch Loss')
  plt.plot(val_trial, val_loss, 'ro', markersize=3, label='Random Validation Batch Loss')
  plt.plot(epoch_trial, epoch_loss, 'bo', markersize=5, label='Validation Loss After 1 Full Epoch')

  # titles and axis labels
  plt.xlabel('Number of Batches')
  plt.ylabel('Loss')
  plt.title('Loss over ' + str(len(epoch_trial)) + ' Epochs')

  plt.legend()
  plt.show() if args.show else plt.savefig(args.plot_name)

def main(args):
  plot(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--attach_train_file', action='append', dest='train_files',
                      default=[],
                      help='File containing training iteration and loss on each line, separated by a comma')
  parser.add_argument('--attach_val_file', action='append', dest='val_files',
                      default=[],
                      help='File containing validation iteration and loss on each line, separated by a comma.')
  parser.add_argument('--plot_name', type=str,
                      default='graph.png',
                      help='Path to save the plot. Default value of graph.png')
  parser.add_argument('-show', action='store_true',
                      default=False,
                      help='Set to display the graph instead of saving it')
  args = parser.parse_args()
  main(args)
