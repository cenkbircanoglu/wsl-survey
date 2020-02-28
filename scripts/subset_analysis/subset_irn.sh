export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export ROOT_FOLDER=./datasets/voc2012/VOCdevkit/VOC2012/

for subset in subset1 subset2 subset3 subset4 subset5 subset6 subset7 subset8 subset9 subset10; do
    export OUTPUT_FOLDER=./outputs/voc12/results/${subset}_$MODEL

    python3 wsl_survey/segmentation/irn/main.py \
        --voc12_root=$ROOT_FOLDER \
        --chainer_eval_set=train \
        --class_label_dict_path=./data/voc12/${subset}/cls_labels.npy \
        --train_list=./data/voc12/${subset}/train_aug.txt \
        --val_list=./data/voc12/${subset}/val.txt \
        --infer_list=./data/voc12/${subset}/train.txt \
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
        --num_workers=8 \
        --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
        --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
        --cam_learning_rate=0.01 \
        --cam_num_epoches=10
done



export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export ROOT_FOLDER=./datasets/voc2012/VOCdevkit/VOC2012/

for subset in subset1 subset2 subset3 subset4 subset5 subset6 subset7 subset8 subset9 subset10; do
    export OUTPUT_FOLDER=./outputs1/voc12/results/${subset}_$MODEL

    python3 wsl_survey/segmentation/irn/main.py \
        --voc12_root=$ROOT_FOLDER \
        --chainer_eval_set=val \
        --class_label_dict_path=./data/voc12/${subset}/cls_labels.npy \
        --train_list=./data/voc12/${subset}/train_aug.txt \
        --val_list=./data/voc12/${subset}/val.txt \
        --infer_list=./data/voc12/${subset}/train.txt \
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
        --cam_to_ir_label_pass=True \
        --num_workers=1 \
        --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
        --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
        --cam_learning_rate=0.01
done
