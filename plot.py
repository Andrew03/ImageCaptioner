import pylab as plt
import csv
import sys

loss = []
trial = [] 
if len(sys.argv) == 1:
    print("Needs a file to plot with!")
    sys.exit()
elif len(sys.argv) == 2:
    with open(sys.argv[1],'r') as file:
        trial_number = 0
        for line in file:
            trial_number = trial_number + 1
            loss.append(str(line))
            trial.append(trial_number)
plt.plot(trial, loss, 'r')
plt.xlabel('trials')
plt.ylabel('loss')
plt.title('loss over trials')
plt.show()


