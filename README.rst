==========
WSL Survey
==========




.. image:: https://pyup.io/repos/github/cenkbircanoglu/wsl_survey/shield.svg
     :target: https://pyup.io/repos/github/cenkbircanoglu/wsl_survey/
     :alt: Updates

[![Build Status](https://travis-ci.com/cenkbircanoglu/wsl-survey.svg?branch=master)]
(https://travis-ci.com/cenkbircanoglu/wsl-survey)


Weakly Supervised Learning Experiments



Features
--------

* TODO


docker run run --rm -i triage/tensorflow-training:latest
  --volume "$(pwd):/trainer"
  --env TRAINING_PYTHON_PACKAGE="/trainer/dist/keras-trainer-0.0.1.tar.gz" \
  --env TRAINING_COMMAND="python -m wsl_survey/wildcat/demo_voc2007 " \
  --env TRAINING_TRAIN_DATASET="/trainer/tests/files/catdog/train.zip" \
  --env TRAINING_VAL_DATASET="/trainer/tests/files/catdog/val.zip" \
  --env TRAINING_OUTPUT_MODEL="/trainer/output/model" \
  --env TRAINING_OUTPUT_LOGS="/trainer/output/logs"
