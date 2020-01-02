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



python3 wsl_survey/acol/evaluate.py \
    --dataset_dir=/Users/cenk.bircanoglu/wsl/wsl_survey/datasets/voc2007/labelled \
    --image_dir=/Users/cenk.bircanoglu/wsl/wsl_survey/datasets/voc2007/images \
    --checkpoints=/Users/cenk.bircanoglu/wsl/wsl_survey/checkpoints/models/acol-voc2007 \
    --epochs=1
