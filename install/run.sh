#!/bin/bash
set -e

if [ -z "$1" ]
  then
    jupyter notebook --allow-root & \
    tensorboard --logdir=logs/ --host=0.0.0.0 --port=6006
elif [ "$1" == *".ipynb"* ]
  then
    jupyter notebook "$1" --allow-root & \
    tensorboard --logdir=logs/ --host=0.0.0.0 --port=6006
else
    exec "$@"
fi

