python3 wsl_survey/datasets/voc/2012/download.py \
    --dataset_dir=/Volumes/2TB/datasets/voc2012

python3 wsl_survey/datasets/voc/preprocess/labelled_dataset.py \
    --dataset_dir=/Volumes/2TB/datasets/voc2012/data \
    --output_dir=/Volumes/2TB/datasets/voc2012/labelled

python3 wsl_survey/datasets/voc/preprocess/annotated_dataset.py \
    --dataset_dir=/Volumes/2TB/datasets/voc2012/data \
    --labelled_dir=/Volumes/2TB/datasets/voc2012/labelled \
    --output_dir=/Volumes/2TB/datasets/voc2012/annotated

python3 wsl_survey/datasets/class_labels.py \
    --source_dir=/Volumes/2TB/datasets/voc2012/labelled \
    --target_dir=/Volumes/2TB/datasets/voc2012/labelled
