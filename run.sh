make dist
make docker-build
make docker-tag
#make docker-release
export IMAGE=cenkbircanoglu/wsl-survey-gpu
export RUNTIME=nvidia
docker-compose -f scripts/irn/voc2007.yml run trainer

#
#make docker-release
#
#docker-compose -f scripts/check_gpu.yml up
#
##IMAGE cenkbircanoglu/wsl-survey cenkbircanoglu/wsl-survey-gpu
#export IMAGE=cenkbircanoglu/wsl-survey-gpu
##RUNTIME runc nvidia
#export RUNTIME=nvidia
#
##DATASET LIST = voc2007 voc2012 coco2014 coco2017
##MODEL LIST = wildcat acol
#
#for DATASET in voc2007 voc2012 coco2014 coco2017; do
#    docker-compose -f scripts/datasets/$DATASET.yml run download_dataset
#    docker-compose -f scripts/datasets/$DATASET.yml run label_dataset
#    docker-compose -f scripts/datasets/$DATASET.yml run annotate_dataset
#    docker-compose -f scripts/datasets/$DATASET.yml run create_class_labels
#    for MODEL in wildcat acol gradcam irn; do
#        docker-compose -f scripts/$MODEL/$DATASET.yml run trainer
#    done
#done
