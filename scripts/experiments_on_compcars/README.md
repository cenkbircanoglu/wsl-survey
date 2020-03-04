

```.env
python3 scripts/experiments_on_compcars/format_data.py
```

```.env
python3 scripts/experiments_on_compcars/random_labels.py
```

### TRAIN USING MAKE ID

```.env
export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export CATEGORY=make
export OUTPUT_FOLDER=./outputs/compcars/${CATEGORY}/results/$MODEL

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
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --train_cam_pass=True \
    --make_cam_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=75
```


### TRAIN USING MODEL ID

```.env
export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export CATEGORY=model
export OUTPUT_FOLDER=./outputs/compcars/${CATEGORY}/results/$MODEL

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
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --train_cam_pass=True \
    --make_cam_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=431
```


### TRAIN USING YEAR ID

```.env
export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export CATEGORY=year
export OUTPUT_FOLDER=./outputs/compcars/${CATEGORY}/results/$MODEL

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
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --train_cam_pass=True \
    --make_cam_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=16
```


### TRAIN USING MAKE YEAR

```.env
export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export CATEGORY=make_year
export OUTPUT_FOLDER=./outputs/compcars/${CATEGORY}/results/$MODEL

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
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --train_cam_pass=True \
    --make_cam_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=492
```

### TRAIN USING MODEL YEAR

```.env
export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export CATEGORY=model_year
export OUTPUT_FOLDER=./outputs/compcars/${CATEGORY}/results/$MODEL

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
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --train_cam_pass=True \
    --make_cam_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=1631
```

