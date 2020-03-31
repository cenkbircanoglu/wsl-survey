export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export CATEGORY=cv_results
for CATEGORY in  make model year make_year kmeans_75 kmeans_431
do
    export OUTPUT_FOLDER=./compcars_outputs/compcars/${CATEGORY}/results/$MODEL
    python3 wsl_survey/segmentation/irn/main.py \
        --class_label_dict_path=./data/compcars/train/cls_labels_make.npy \
        --voc12_root=./data/compcars/ \
        --train_list=./data/compcars/train/test.txt \
        --val_list=./data/compcars/train/test.txt \
        --infer_list=./data/compcars/train/test.txt \
        --cam_weights_name=$OUTPUT_FOLDER/sess/cam.pth \
        --irn_weights_name=$OUTPUT_FOLDER/sess/irn.pth \
        --cam_out_dir=$OUTPUT_FOLDER/cam \
        --sem_seg_out_dir=$OUTPUT_FOLDER/sem_seg \
        --ins_seg_out_dir=$OUTPUT_FOLDER/ins_seg \
        --ir_label_out_dir=$OUTPUT_FOLDER/irn_label \
        --bbox_out_dir=$OUTPUT_FOLDER/bbox \
        --cam_network=$NETWORK \
        --irn_network=$NETWORK \
        --log_name=$OUTPUT_FOLDER/logs \
        --eval_bbox_pass=True \
        --num_workers=64 \
        --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
        --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
        --cam_batch_size=16 \
        --irn_batch_size=16 \
        --num_classes=20

done
