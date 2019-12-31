export IMAGE=cenkbircanoglu/wsl-survey-gpu
export RUNTIME=nvidia

docker-compose -f scripts/wildcat/voc2007.yml run trainer

docker-compose -f scripts/acol/voc2007.yml run trainer

docker-compose -f scripts/gradcam/voc2007.yml run trainer

docker-compose -f scripts/irn/voc2007.yml run trainer
