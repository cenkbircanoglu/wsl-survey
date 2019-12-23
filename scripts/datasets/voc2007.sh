python3 wsl_survey/datasets/voc/2007/download.py \
    --dataset_dir=datasets/voc2007

python3 wsl_survey/datasets/voc/preprocess/labelled_dataset.py \
    --dataset_dir=datasets/voc2007/data \
    --output_dir=datasets/voc2007/labelled

python3 wsl_survey/datasets/voc/preprocess/annotated_dataset.py \
    --dataset_dir=datasets/voc2007/data \
    --labelled_dir=datasets/voc2007/labelled \
    --output_dir=datasets/voc2007/annotated

python3 wsl_survey/datasets/class_labels.py \
    --source_dir=datasets/voc2007/labelled \
    --target_dir=datasets/voc2007/labelled
