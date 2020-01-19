export PYTHONPATH='.'

#python3 wsl_survey/segmentation/irn/step/train_cam.py
#python3 wsl_survey/segmentation/irn/step/make_cam.py
#python3 wsl_survey/segmentation/irn/step/eval_cam.py
#python3 wsl_survey/segmentation/irn/step/cam_to_ir_label.py
#python3 wsl_survey/segmentation/irn/step/train_irn.py
#python3 wsl_survey/segmentation/irn/step/make_sem_seg_labels.py
#python3 wsl_survey/segmentation/irn/step/eval_sem_seg.py
#python3 wsl_survey/segmentation/irn/step/make_ins_seg_labels.py
#python3 wsl_survey/segmentation/irn/step/eval_ins_seg.py

###### TRAIN and INFERENCE ######

export MODEL=resnet18
export NETWORK=ResNet18
export ROOT_FOLDER=./data/test1/VOC2012
export SEGMENTATION_DATA_FOLDER=./data/test1/VOC2012/ImageSets/Segmentation
export OUTPUT_FOLDER=./outputs/test1/results/$MODEL/

python3 wsl_survey/segmentation/irn/main.py \
    --voc12_root=$ROOT_FOLDER \
    --chainer_eval_set=val \
    --class_label_dict_path=./data/voc12/cls_labels.npy \
    --train_list=$SEGMENTATION_DATA_FOLDER/train_aug.txt \
    --val_list=$SEGMENTATION_DATA_FOLDER/val.txt \
    --infer_list=$SEGMENTATION_DATA_FOLDER/val.txt \
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
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=4 \
    --irn_batch_size=4


###### DISTILLED TRAIN and INFERENCE ######


export MODEL=resnet50
export NETWORK=ResNet50
export ROOT_FOLDER=./data/test1/VOC2012
export SEGMENTATION_DATA_FOLDER=./data/test1/VOC2012/ImageSets/Segmentation
export OUTPUT_FOLDER=./outputs/test1/results/distilled_$MODEL/

python3 wsl_survey/segmentation/irn/main.py \
    --voc12_root=$ROOT_FOLDER \
    --chainer_eval_set=val \
    --class_label_dict_path=./data/voc12/cls_labels.npy \
    --train_list=$SEGMENTATION_DATA_FOLDER/train_aug.txt \
    --val_list=$SEGMENTATION_DATA_FOLDER/val.txt \
    --infer_list=$SEGMENTATION_DATA_FOLDER/val.txt \
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
    --cam_network_module=wsl_survey.segmentation.irn.net.distilled.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.distilled.resnet_irn \
    --cam_batch_size=4 \
    --irn_batch_size=4
