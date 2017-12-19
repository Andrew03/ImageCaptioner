import pylab as plt
import matplotlib.patches as mpatches
import csv
import sys

train_loss = []
train_trial = [] 
val_loss = []
val_trial = [] 
epoch_loss = []
epoch_trial = [] 
if len(sys.argv) == 1:
  print("Needs a file to plot with!")
  sys.exit()
elif len(sys.argv) == 2:
  with open(sys.argv[1],'r') as file:
    trial_number = 0
    for line in file:
      number_value = line.split(",")
      train_loss.append(number_value[1])
      train_trial.append(number_value[0])
else:
  for i in range(1, len(sys.argv)):
    with open(sys.argv[i], 'r') as file:
      endOfEpoch = False
      for line in file:
        number_value = line.split(",")
        if i < (len(sys.argv) + 1) / 2:
          train_loss.append(number_value[1])
          train_trial.append(number_value[0])
        else:
          if endOfEpoch == True:
            epoch_loss.append(number_value[1])
            epoch_trial.append(number_value[0])
            endOfEpoch = False
          else:
            if len(number_value) == 1:
              endOfEpoch = True
            else:
              val_loss.append(number_value[1])
              val_trial.append(number_value[0])

# plotting the data
plt.plot(train_trial, train_loss, 'go', markersize=1, label='Training Batch Loss')
plt.plot(val_trial, val_loss, 'ro', markersize=3, label='Random Validation Batch Loss')
plt.plot(epoch_trial, epoch_loss, 'bo', markersize=5, label='Validation Loss After 1 Full Epoch')

# titles and axis labels
plt.xlabel('Number of Batches')
plt.ylabel('Loss')
plt.title('Loss over ' + str(len(epoch_trial)) + ' Epochs')

plt.legend()

plt.show()
