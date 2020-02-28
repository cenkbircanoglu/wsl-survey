export PYTHONPATH='.'

export MODEL=deeplabv3
export NETWORK=DeepLabV3
export ROOT_FOLDER=./datasets/voc2012/VOCdevkit/VOC2012/
export SEGMENTATION_DATA_FOLDER=./data/test1/VOC2012/ImageSets/Segmentation
export OUTPUT_FOLDER=./outputs/voc12/results/$MODEL/

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
    --train_irn_pass=True \
    --make_sem_seg_pass=True \
    --eval_sem_seg_pass=True \
    --num_workers=8 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=8 \
    --irn_batch_size=8



#./outputs/voc12/results/deeplabv3//cam
#{'iou': array([0.80563274, 0.46921445, 0.29454158, 0.44945024, 0.39209913,
#       0.46937367, 0.63886961, 0.57308867, 0.51145128, 0.35803821,
#       0.56989898, 0.5107898 , 0.57619841, 0.5042012 , 0.60156954,
#       0.56340369, 0.41434724, 0.62366423, 0.4862939 , 0.54635895,
#      0.49153061]), 'miou': 0.5166674351523047}

eval_sem_seg
step.eval_sem_seg: Sun Feb 23 22:06:03 2020
100%|##################################################################################################################################################################| 1464/1464 [00:01<00:00, 831.08it/s]
0.07893536303693238 0.057977050002102626
0.17561533048767142 0.2190024929583007
{'iou': array([0.86308759, 0.71945067, 0.34289008, 0.6221479 , 0.47205865,
       0.54871329, 0.70513799, 0.60942963, 0.67669335, 0.39647979,
       0.73153099, 0.46543966, 0.72993955, 0.69755819, 0.66090274,
       0.65726992, 0.45950848, 0.78306904, 0.5689836 , 0.64135659,
       0.61908344]), 'miou': 0.617653862763882}
