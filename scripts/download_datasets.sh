docker-compose -f scripts/datasets/voc2007.yml run download_dataset
docker-compose -f scripts/datasets/voc2007.yml run label_dataset
docker-compose -f scripts/datasets/voc2007.yml run annotate_dataset
docker-compose -f scripts/datasets/voc2007.yml run create_class_labels

docker-compose -f scripts/datasets/voc2012.yml run download_dataset
docker-compose -f scripts/datasets/voc2012.yml run label_dataset
docker-compose -f scripts/datasets/voc2012.yml run annotate_dataset
docker-compose -f scripts/datasets/voc2012.yml run create_class_labels

docker-compose -f scripts/datasets/coco2014.yml run download_dataset
docker-compose -f scripts/datasets/coco2014.yml run label_dataset
docker-compose -f scripts/datasets/coco2014.yml run annotate_dataset
docker-compose -f scripts/datasets/coco2014.yml run create_class_labels

docker-compose -f scripts/datasets/coco2017.yml run download_dataset
docker-compose -f scripts/datasets/coco2017.yml run label_dataset
docker-compose -f scripts/datasets/coco2017.yml run annotate_dataset
docker-compose -f scripts/datasets/coco2017.yml run create_class_labels
