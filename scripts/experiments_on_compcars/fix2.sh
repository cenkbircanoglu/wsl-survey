export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export CATEGORY=make
export OUTPUT_FOLDER=./compcars_outputs/compcars/${CATEGORY}/results/$MODEL

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
    --voc12_root=./data/compcars/ \
    --train_list=./data/compcars/train/train.txt \
    --val_list=./data/compcars/train/test.txt \
    --infer_list=./data/compcars/train/train.txt \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=75

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --eval_bbox_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=75


export CATEGORY=make_year
export OUTPUT_FOLDER=./compcars_outputs/compcars/${CATEGORY}/results/$MODEL

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
    --voc12_root=./data/compcars/ \
    --train_list=./data/compcars/train/train.txt \
    --val_list=./data/compcars/train/test.txt \
    --infer_list=./data/compcars/train/train.txt \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=492

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --eval_bbox_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=492


export CATEGORY=model
export OUTPUT_FOLDER=./compcars_outputs/compcars/${CATEGORY}/results/$MODEL

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
    --voc12_root=./data/compcars/ \
    --train_list=./data/compcars/train/train.txt \
    --val_list=./data/compcars/train/test.txt \
    --infer_list=./data/compcars/train/train.txt \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=431

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --eval_bbox_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=431


export CATEGORY=model_year
export OUTPUT_FOLDER=./compcars_outputs/compcars/${CATEGORY}/results/$MODEL

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
    --voc12_root=./data/compcars/ \
    --train_list=./data/compcars/train/train.txt \
    --val_list=./data/compcars/train/test.txt \
    --infer_list=./data/compcars/train/train.txt \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=1631

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --eval_bbox_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=1631

export CATEGORY=random_100
export OUTPUT_FOLDER=./compcars_outputs/compcars/${CATEGORY}/results/$MODEL

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
    --voc12_root=./data/compcars/ \
    --train_list=./data/compcars/train/train.txt \
    --val_list=./data/compcars/train/test.txt \
    --infer_list=./data/compcars/train/train.txt \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=100

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --eval_bbox_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=100

export CATEGORY=year
export OUTPUT_FOLDER=./compcars_outputs/compcars/${CATEGORY}/results/$MODEL

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
    --voc12_root=./data/compcars/ \
    --train_list=./data/compcars/train/train.txt \
    --val_list=./data/compcars/train/test.txt \
    --infer_list=./data/compcars/train/train.txt \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=16

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/compcars/train/cls_labels_${CATEGORY}.npy \
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
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --eval_bbox_pass=True \
    --num_workers=4 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=1 \
    --irn_batch_size=1 \
    --num_classes=16

