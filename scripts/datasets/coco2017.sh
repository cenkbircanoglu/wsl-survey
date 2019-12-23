python3 wsl_survey/datasets/coco/2017/download.py \
    --dataset_dir=datasets/coco2017

python3 wsl_survey/datasets/coco/preprocess/labelled_dataset.py \
    --dataset_dir=datasets/coco2017/annotations \
    --output_dir=datasets/coco2017/labelled

python3 wsl_survey/datasets/coco/preprocess/annotated_dataset.py \
    --dataset_dir=datasets/coco2017/annotations \
    --output_dir=datasets/coco2017/annotated

python3 wsl_survey/datasets/class_labels.py \
    --source_dir=datasets/coco2017/labelled \
    --target_dir=datasets/coco2017/labelled
