

make docker-release

docker run -i -t --rm -e TRAINING_COMMAND='python -c "import torch;print(torch.cuda.is_available())"'  cenkbircanoglu/wsl-survey-gpu

docker-compose -f scripts/experiments/wildcat/cpu.yml up

docker-compose -f scripts/experiments/wildcat/gpu.yml up

docker-compose -f scripts/experiments/check_gpu.yml up



$(python -c "import torch;print(torch.cuda.is_available())")
