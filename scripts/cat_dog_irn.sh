export PYTHONPATH='.'

export MODEL=resnet152
export NETWORK=ResNet152
export ROOT_FOLDER=./datasets/voc2012/VOCdevkit/VOC2012/
export SEGMENTATION_DATA_FOLDER=./data/test1/VOC2012/ImageSets/Segmentation
export OUTPUT_FOLDER=./outputs/voc12/results/cat_dog_$MODEL

python3 wsl_survey/segmentation/irn/main.py \
    --voc12_root=$ROOT_FOLDER \
    --chainer_eval_set=train \
    --class_label_dict_path=./data/voc12/cat_dog/cls_labels.npy \
    --train_list=./data/voc12/cat_dog/train_aug.txt \
    --val_list=./data/voc12/cat_dog/val.txt \
    --infer_list=./data/voc12/cat_dog/train.txt \
    --cam_weights_name=$OUTPUT_FOLDER/sess/cam.pth \
    --irn_weights_name=$OUTPUT_FOLDER/sess/irn.pth \
    --cam_out_dir=$OUTPUT_FOLDER/cam \
    --sem_seg_out_dir=$OUTPUT_FOLDER/sem_seg \
    --ins_seg_out_dir=$OUTPUT_FOLDER/ins_seg \
    --ir_label_out_dir=$OUTPUT_FOLDER/irn_label \
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --eval_cam_pass=True \
    --num_workers=1 \
    --cam_network_module=wsl_survey.segmentation.irn.net.distilled.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.distilled.resnet_irn
