#!/bin/bash

if [ $# -ne 1 ]; then 
  echo "Need to specify min_occurrences and batch size"
  exit 0
else
  FILES="$(python ./notifications/getOutputName.py $1)"
  arr=($FILES)
  echo ${arr[0]}
  echo ${arr[1]}
fi
