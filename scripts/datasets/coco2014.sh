python3 wsl_survey/datasets/coco/2014/download.py \
    --dataset_dir=datasets/coco2014

python3 wsl_survey/datasets/coco/preprocess/labelled_dataset.py \
    --dataset_dir=datasets/coco2014/annotations \
    --output_dir=datasets/coco2014/labelled

python3 wsl_survey/datasets/coco/preprocess/annotated_dataset.py \
    --dataset_dir=datasets/coco2014/annotations \
    --output_dir=datasets/coco2014/annotated

python3 wsl_survey/datasets/class_labels.py \
    --source_dir=datasets/coco2014/labelled \
    --target_dir=datasets/coco2014/labelled
