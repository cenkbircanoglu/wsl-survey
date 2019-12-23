

make docker-release

nvidia-docker run -i -t --rm \
    -e TRAINING_COMMAND='python -c "import torch;print(torch.cuda.is_available())"'  cenkbircanoglu/wsl-survey-gpu

docker-compose -f scripts/experiments/wildcat/cpu.yml up

nvidia-docker run -i -t --rm \
    -e TRAINING_COMMAND='python /app/wsl_survey/wildcat/demo_voc2007.py dataset' cenkbircanoglu/wsl-survey-gpu
