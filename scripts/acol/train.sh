python3 wsl_survey/acol/main.py \
    --dataset_dir=datasets/test/labelled \
    --image_dir=datasets/test/images \
    --checkpoints=checkpoints/acol-test \
    --epochs=1


python3 wsl_survey/acol/evaluate.py \
    --dataset_dir=datasets/test/labelled \
    --image_dir=datasets/test/images \
    --checkpoints=checkpoints/acol-test \
    --epochs=1
