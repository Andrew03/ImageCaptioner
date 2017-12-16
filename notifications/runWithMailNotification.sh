#!/bin/bash

OPEN_GPU="$(./notifications/findOpenGPU.sh)"
if [ $# -eq 0 ]; then
  echo "Need to enter a program, parameter file, and email to send to"
  exit 0
fi
if [ $OPEN_GPU -eq -1 ]; then
  echo "No GPUs are open"
  exit
fi

OUTPUT_NAMES="$(./notifications/getOutputName.sh $2)"
OUTPUT_NAMES=($OUTPUT_NAMES)
TRAIN_FILE=${OUTPUT_NAMES[0]}
VAL_FILE=${OUTPUT_NAMES[1]}
CUDA_VISIBLE_DEVICES=$OPEN_GPU python $1 $2

if [ $? -eq 0 ]; then
  mail -a $TRAIN_FILE -a $VAL_FILE -s "Run Suceeded" $3 < $2
else
  mail -s "Run Failed" $3 < $2
fi
