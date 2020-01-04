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
    --dataset_dir=datasets/voc2007/labelled \
    --image_dir=datasets/voc2007/data/JPEGImages \
    --arch=vgg_v1  \
    --checkpoints=checkpoints/models/acol-voc2007 \
    --output=checkpoints/models/acol-voc2007/outputs

python3 wsl_survey/acol/evaluate.py \
    --dataset_dir=datasets/voc2012/labelled \
    --image_dir=datasets/voc2012/data/JPEGImages \
    --arch=vgg_v1  \
    --checkpoints=checkpoints/models/acol-voc2012 \
    --output=checkpoints/models/acol-voc2012/outputs
