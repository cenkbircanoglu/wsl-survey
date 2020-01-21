export PYTHONPATH='.'

export MODEL=resnet152
export NETWORK=ResNet152
export ROOT_FOLDER=./datasets/voc2012/VOCdevkit/VOC2012/
export SEGMENTATION_DATA_FOLDER=./data/test1/VOC2012/ImageSets/Segmentation
export OUTPUT_FOLDER=./outputs/voc12/results/distilled_more_$MODEL/

python3 wsl_survey/segmentation/irn/main.py \
    --voc12_root=$ROOT_FOLDER \
    --chainer_eval_set=train \
    --class_label_dict_path=./data/voc12/cls_labels.npy \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/val.txt \
    --infer_list=./data/voc12/train.txt \
    --cam_weights_name=$OUTPUT_FOLDER/sess/cam.pth \
    --irn_weights_name=$OUTPUT_FOLDER/sess/irn.pth \
    --cam_out_dir=$OUTPUT_FOLDER/cam \
    --sem_seg_out_dir=$OUTPUT_FOLDER/sem_seg \
    --ins_seg_out_dir=$OUTPUT_FOLDER/ins_seg \
    --ir_label_out_dir=$OUTPUT_FOLDER/irn_label \
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --train_cam_pass=True \
    --make_cam_pass=True \
    --eval_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --train_irn_pass=True \
    --make_sem_seg_pass=True \
    --eval_sem_seg_pass=True \
    --num_workers=1 \
    --cam_network_module=wsl_survey.segmentation.irn.net.distilled_more.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.distilled_more.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16



python3 wsl_survey/segmentation/irn/main.py \
    --voc12_root=$ROOT_FOLDER \
    --chainer_eval_set=val \
    --class_label_dict_path=./data/voc12/cls_labels.npy \
    --train_list=./data/voc12/val.txt \
    --val_list=./data/voc12/val.txt \
    --infer_list=./data/voc12/val.txt \
    --cam_weights_name=$OUTPUT_FOLDER/sess/cam.pth \
    --irn_weights_name=$OUTPUT_FOLDER/sess/irn.pth \
    --cam_out_dir=$OUTPUT_FOLDER/cam \
    --sem_seg_out_dir=$OUTPUT_FOLDER/sem_seg \
    --ins_seg_out_dir=$OUTPUT_FOLDER/ins_seg \
    --ir_label_out_dir=$OUTPUT_FOLDER/irn_label \
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --make_cam_pass=True \
    --eval_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --make_sem_seg_pass=True \
    --eval_sem_seg_pass=True \
    --num_workers=1 \
    --cam_network_module=wsl_survey.segmentation.irn.net.distilled_more.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.distilled_more.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16
