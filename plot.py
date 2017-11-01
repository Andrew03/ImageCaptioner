import pylab as plt
import csv
import sys

train_loss = []
train_trial = [] 
val_loss = []
val_trial = [] 
if len(sys.argv) == 1:
  print("Needs a file to plot with!")
  sys.exit()
elif len(sys.argv) == 2:
  with open(sys.argv[1],'r') as file:
    trial_number = 0
    index = 0
    for line in file:
      if index % 2 == 0:
        trial_number = trial_number + 1
        train_loss.append(str(line))
        train_trial.append(trial_number)
      else: 
        val_loss.append(str(line))
        val_trial.append(trial_number)
plt.plot(train_trial, train_loss, 'ro')
plt.plot(val_trial, val_loss, 'bo')
plt.xlabel('trials')
plt.ylabel('loss')
plt.title('loss over trials')
plt.show()


