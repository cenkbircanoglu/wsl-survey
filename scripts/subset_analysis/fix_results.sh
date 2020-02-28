
for subset in subset1 subset2 subset3 subset4 subset5 subset6 subset7 subset8 subset9 subset19; do
    python3 wsl_survey/segmentation/irn/step/eval_cam.py \
        --cam_out_dir=./results/${subset}_resnet152/${subset}_resnet152/cam \
        --chainer_eval_set=train
done


for subset in subset1 subset2 subset3 subset4 subset5 subset6 subset7 subset8 subset9 subset19; do
    python3 wsl_survey/segmentation/irn/step/eval_cam.py \
        --cam_out_dir=./results/${subset}_resnet152/${subset}_resnet152/cam \
        --chainer_eval_set=val
done
