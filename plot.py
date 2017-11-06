import pylab as plt
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
    index = 0
    for line in file:
      number_value = line.split(",")
      train_loss.append(number_value[1])
      train_trial.append(number_value[0])
elif len(sys.argv) == 3:
  with open(sys.argv[1],'r') as file:
    index = 0
    for line in file:
      number_value = line.split(",")
      train_loss.append(number_value[1])
      train_trial.append(number_value[0])
  with open(sys.argv[2],'r') as file:
    index = 0
    endOfEpoch = False
    for line in file:
      if endOfEpoch == True:
        number_value = line.split(",")
        epoch_loss.append(number_value[1])
        epoch_trial.append(number_value[0])
        endOfEpoch = False
      else:
        number_value = line.split(",")
        if len(number_value) == 1:
          endOfEpoch = True
        else:
          val_loss.append(number_value[1])
          val_trial.append(number_value[0])

plt.plot(train_trial, train_loss, 'ro', markerSize=1)
plt.plot(val_trial, val_loss, 'bo', markerSize=5)
plt.plot(epoch_trial, epoch_loss, 'bo', markerSize=10)
plt.xlabel('trials')
plt.ylabel('loss')
plt.title('loss over trials')
plt.show()


