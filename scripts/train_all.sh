export IMAGE=cenkbircanoglu/wsl-survey-gpu
export RUNTIME=nvidia

docker-compose -f scripts/wildcat/voc2007.yml run trainer
docker-compose -f scripts/wildcat/voc2012.yml run trainer
docker-compose -f scripts/wildcat/coco2014.yml run trainer
docker-compose -f scripts/wildcat/coco2017.yml run trainer

docker-compose -f scripts/acol/voc2007.yml run trainer
docker-compose -f scripts/acol/voc2012.yml run trainer
docker-compose -f scripts/acol/coco2014.yml run trainer
docker-compose -f scripts/acol/coco2017.yml run trainer

docker-compose -f scripts/gradcam/voc2007.yml run trainer
docker-compose -f scripts/gradcam/voc2012.yml run trainer
docker-compose -f scripts/gradcam/coco2014.yml run trainer
docker-compose -f scripts/gradcam/coco2017.yml run trainer

docker-compose -f scripts/irn/voc2007.yml run trainer
docker-compose -f scripts/irn/voc2012.yml run trainer
docker-compose -f scripts/irn/coco2014.yml run trainer
docker-compose -f scripts/irn/coco2017.yml run trainer
