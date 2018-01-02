#!/bin/bash

nvidia-smi -i 0 > in_use0.txt
nvidia-smi -i 1 > in_use1.txt
nvidia-smi -i 2 > in_use2.txt
IN_USE_0="$(grep 'No running processes found' in_use0.txt)"
IN_USE_1="$(grep 'No running processes found' in_use1.txt)"
IN_USE_2="$(grep 'No running processes found' in_use2.txt)"
OPEN=0
if [ "$IN_USE_0" != '' ]; then
  OPEN=0
elif [ "$IN_USE_1" != '' ]; then
  OPEN=1
elif [ "$IN_USE_2" != '' ]; then
  OPEN=2
fi
rm in_use0.txt
rm in_use1.txt
rm in_use2.txt
echo $OPEN
