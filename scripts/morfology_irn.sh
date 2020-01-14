export MODEL=resnet152

python3 wsl_survey/segmentation/irn/morph/apply_morph_ir_label.py --kernel_size=3

for MORP in eroded dilated opened closed gaussian
do
python3 wsl_survey/segmentation/irn/main.py \
    --voc12_root=./datasets/voc2012/VOCdevkit/VOC2012 \
    --chainer_eval_set=train \
    --class_label_dict_path=./data/voc12/cls_labels.npy \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/val.txt \
    --infer_list=./data/voc12/train.txt \
    --cam_weights_name=./outputs/voc12/results//$MODEL/sess/cam.pth \
    --irn_weights_name=./outputs/voc12/results/$MODEL/sess/irn.pth \
    --cam_out_dir=./outputs/voc12/results/$MODEL/cam \
    --sem_seg_out_dir=./outputs/voc12/results/$MODEL/sem_seg \
    --ins_seg_out_dir=./outputs/voc12/results/$MODEL/ins_seg \
    --ir_label_out_dir=./outputs/voc12/results/$MODEL/irn_label_${MORP} \
    --cam_network=net.${MODEL}_cam \
    --irn_network=net.${MODEL}_irn \
    --log_name=./outputs/voc12/logs/$MODEL \
    --train_irn_pass=True \
    --make_ins_seg_pass=True \
    --eval_ins_seg_pass=True \
    --make_sem_seg_pass=True \
    --eval_sem_seg_pass=True \
    --num_workers=8 \
    --cam_batch_size=8 \
    --irn_batch_size=8 &
done