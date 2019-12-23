
nvidia-docker run -i -t --rm \
    -e TRAINING_COMMAND='python -c "import torch;print(torch.cuda.is_available())"'  cenkbircanoglu/wsl-survey-gpu
