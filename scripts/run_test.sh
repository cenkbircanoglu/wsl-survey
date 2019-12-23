python3 wsl_survey/datasets/class_labels.py \
    --source_dir=datasets/test/labelled \
    --target_dir=datasets/test/labelled

sh scripts/wildcat/train.sh
sh scripts/acol/train.sh
sh scripts/gradcam/train.sh
sh scripts/irn/train.sh

