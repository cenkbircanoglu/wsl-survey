export IMAGE=cenkbircanoglu/wsl-survey-gpu
export RUNTIME=nvidia

docker-compose -f scripts/wildcat/voc2007.yml run trainer > logs/voc2007_wildcat.log
docker-compose -f scripts/acol/voc2007.yml run trainer > logs/voc2007_acol.log
docker-compose -f scripts/acol/voc2007.yml run evaluator > logs/voc2007_acol_eval.log
docker-compose -f scripts/gradcam/voc2007.yml run trainer > logs/voc2007_gradcam.log
docker-compose -f scripts/gradcam/voc2007.yml run evaluator > logs/voc2007_gradcam_eval.log
docker-compose -f scripts/irn/voc2007.yml run trainer > logs/voc2007_irn.log
