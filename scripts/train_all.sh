export IMAGE=cenkbircanoglu/wsl-survey-gpu
export RUNTIME=nvidia

mkdir logs

docker-compose -f scripts/wildcat/voc2007.yml run trainer > logs/voc2007_wildcat.log
docker-compose -f scripts/acol/voc2007.yml run trainer > logs/voc2007_acol.log
docker-compose -f scripts/acol/voc2007.yml run evaluator > logs/voc2007_acol_eval.log
docker-compose -f scripts/gradcam/voc2007.yml run trainer > logs/voc2007_gradcam.log
docker-compose -f scripts/gradcam/voc2007.yml run evaluator > logs/voc2007_gradcam_eval.log
docker-compose -f scripts/irn/voc2007.yml run trainer > logs/voc2007_irn.log

docker-compose -f scripts/wildcat/voc2012.yml run trainer > logs/voc2012_wildcat.log
docker-compose -f scripts/acol/voc2012.yml run trainer > logs/voc2012_acol.log
docker-compose -f scripts/acol/voc2012.yml run evaluator > logs/voc2012_acol_eval.log
docker-compose -f scripts/gradcam/voc2012.yml run trainer > logs/voc2012_gradcam.log
docker-compose -f scripts/gradcam/voc2012.yml run evaluator > logs/voc2012_gradcam_eval.log
docker-compose -f scripts/irn/voc2012.yml run trainer > logs/voc2012_irn.log

docker-compose -f scripts/wildcat/coco2014.yml run trainer > logs/coco2014_wildcat.log
docker-compose -f scripts/acol/coco2014.yml run trainer > logs/coco2014_acol.log
docker-compose -f scripts/acol/coco2014.yml run evaluator > logs/coco2014_acol_eval.log
docker-compose -f scripts/gradcam/coco2014.yml run trainer > logs/coco2014_gradcam.log
docker-compose -f scripts/gradcam/coco2014.yml run evaluator > logs/coco2014_gradcam_eval.log
docker-compose -f scripts/irn/coco2014.yml run trainer > logs/coco2014_irn.log

docker-compose -f scripts/wildcat/coco2017.yml run trainer > logs/coco2017_wildcat.log
docker-compose -f scripts/acol/coco2017.yml run trainer > logs/coco2017_acol.log
docker-compose -f scripts/acol/coco2017.yml run evaluator > logs/coco2017_acol_eval.log
docker-compose -f scripts/gradcam/coco2017.yml run trainer > logs/coco2017_gradcam.log
docker-compose -f scripts/gradcam/coco2017.yml run evaluator > logs/coco2017_gradcam_eval.log
docker-compose -f scripts/irn/coco2017.yml run trainer > logs/coco2017_irn.log
