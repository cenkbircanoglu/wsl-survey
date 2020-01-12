
###### TRAIN and TRAIN INFERENCE ######
export MODEL=resnet50
python3 wsl_survey/segmentation/irn/main.py \
    --voc12_root=/Volumes/2TB/datasets/voc2012/tmp/VOCdevkit/VOC2012 \
    --chainer_eval_set=train \
    --class_label_dict_path=./data/voc12/cls_labels.npy \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/val.txt \
    --infer_list=./data/voc12/train.txt \
    --cam_weights_name=/Volumes/2TB/voc12/outputs/voc12/results/$MODEL/sess/cam.pth \
    --irn_weights_name=./outputs/voc12/results/$MODEL/sess/irn.pth \
    --cam_out_dir=./outputs/voc12/results/$MODEL/cam \
    --sem_seg_out_dir=./outputs/voc12/results/$MODEL/sem_seg \
    --ins_seg_out_dir=./outputs/voc12/results/$MODEL/ins_seg \
    --ir_label_out_dir=./outputs/voc12/results/$MODEL/irn_label \
    --cam_network=net.${MODEL}_cam \
    --irn_network=net.${MODEL}_irn \
    --log_name=./outputs/voc12/logs/$MODEL \
    --make_cam_pass=True \
    --num_workers=8


for MORP in grayscaled, eroded, dilated, opened, hull1; do
  python3 wsl_survey/segmentation/irn/main.py \
      --voc12_root=/Volumes/2TB/datasets/voc2012/tmp/VOCdevkit/VOC2012 \
      --chainer_eval_set=train \
      --class_label_dict_path=./data/voc12/cls_labels.npy \
      --train_list=./data/voc12/train_aug.txt \
      --val_list=./data/voc12/val.txt \
      --infer_list=./data/voc12/train.txt \
      --cam_weights_name=/Volumes/2TB/voc12/outputs/voc12/results/$MODEL/sess/cam.pth \
      --irn_weights_name=./outputs/voc12/results/$MODEL/sess/irn.pth \
      --cam_out_dir=./outputs/voc12/results/$MODEL/cam/MORP \
      --sem_seg_out_dir=./outputs/voc12/results/$MODEL/sem_seg \
      --ins_seg_out_dir=./outputs/voc12/results/$MODEL/ins_seg \
      --ir_label_out_dir=./outputs/voc12/results/$MODEL/irn_label \
      --cam_network=net.${MODEL}_cam \
      --irn_network=net.${MODEL}_irn \
      --log_name=./outputs/voc12/logs/$MODEL \
      --eval_cam_pass=True \
      --num_workers=4
done
