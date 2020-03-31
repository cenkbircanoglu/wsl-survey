export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export OUTPUT_FOLDER=./outputs/disfa/conf/results/$MODEL

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/disfa/cls_labels_conf.npy \
    --voc12_root=./data/disfa/ \
    --train_list=./data/disfa/train.txt \
    --val_list=./data/disfa/test.txt \
    --infer_list=./data/disfa/train.txt \
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
    --train_cam_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=58


python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/disfa/cls_labels_conf.npy \
    --voc12_root=./data/disfa/ \
    --train_list=./data/disfa/train.txt \
    --val_list=./data/disfa/test.txt \
    --infer_list=./data/disfa/train.txt \
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
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=58


python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/disfa/cls_labels_conf.npy \
    --voc12_root=./data/disfa/ \
    --train_list=./data/disfa/train.txt \
    --val_list=./data/disfa/test.txt \
    --infer_list=./data/disfa/train.txt \
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
    --cam_to_ir_label_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=58

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/disfa/cls_labels_conf.npy \
    --voc12_root=./data/disfa/ \
    --train_list=./data/disfa/train.txt \
    --val_list=./data/disfa/test.txt \
    --infer_list=./data/disfa/train.txt \
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
    --train_irn_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=58

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/disfa/cls_labels_conf.npy \
    --voc12_root=./data/disfa/ \
    --train_list=./data/disfa/train.txt \
    --val_list=./data/disfa/test.txt \
    --infer_list=./data/disfa/train.txt \
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
    --make_sem_seg_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=58



python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/disfa/cls_labels_conf.npy \
    --voc12_root=./data/disfa/ \
    --train_list=./data/disfa/test.txt \
    --val_list=./data/disfa/test.txt \
    --infer_list=./data/disfa/test.txt \
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
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=58



python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/disfa/cls_labels_conf.npy \
    --voc12_root=./data/disfa/ \
    --train_list=./data/disfa/test.txt \
    --val_list=./data/disfa/test.txt \
    --infer_list=./data/disfa/test.txt \
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
    --cam_to_ir_label_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=58

python3 wsl_survey/segmentation/irn/main.py \
    --class_label_dict_path=./data/disfa/cls_labels_conf.npy \
    --voc12_root=./data/disfa/ \
    --train_list=./data/disfa/test.txt \
    --val_list=./data/disfa/test.txt \
    --infer_list=./data/disfa/test.txt \
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
    --make_sem_seg_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16 \
    --num_classes=58
