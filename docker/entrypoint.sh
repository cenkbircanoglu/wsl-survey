#!/usr/bin/env bash
set -e

export PYTHONPATH='/app'
eval $COMMAND
#python /app/wsl_survey/wildcat/demo_voc2007.py dataset

## Fetch inputs
#stored sync $TRAINING_TRAIN_DATASET /training/dataset/train
#stored sync $TRAINING_VAL_DATASET /training/dataset/val
#
## Run training
#echo $TRAINING_COMMAND > /tmp/base64_command
#eval $(base64 -d /tmp/base64_command)
#
## Store outputs
#stored sync /training/output/model $TRAINING_OUTPUT_MODEL
#stored sync /training/output/logs $TRAINING_OUTPUT_LOGS
