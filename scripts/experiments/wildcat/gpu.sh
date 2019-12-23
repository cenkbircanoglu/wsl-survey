
nvidia-docker run -i -t --rm \
    -e TRAINING_COMMAND='python /app/wsl_survey/wildcat/demo_voc2007.py dataset' cenkbircanoglu/wsl-survey-gpu
