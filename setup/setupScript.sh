#!/bin/bash

if [ $# -eq 2 ]; then
  DIRNAME=`dirname $0`
  python $DIRNAME"/setup.py" $1 $2
else
  echo "Need to specify min_occurrences and batch size"
  exit 0
fi
