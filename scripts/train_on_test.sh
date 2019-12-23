export IMAGE=cenkbircanoglu/wsl-survey-gpu
export RUNTIME=nvidia

docker-compose -f scripts/wildcat/test.yml run trainer
docker-compose -f scripts/acol/test.yml run trainer
docker-compose -f scripts/gradcam/test.yml run trainer
docker-compose -f scripts/irn/test.yml run trainer
