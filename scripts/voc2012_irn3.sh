
###### TEST INFERENCE ######
for MODEL in resnet18 resnet34 resnet50 resnet101 resnext50 resnext101 wide_resnet50 wide_resnet101; do
    export MODEL=resnet50
    python3 wsl_survey/segmentation/irn/main.py \
        --voc12_root=./datasets/voc2012/VOCdevkit/VOC2012 \
        --chainer_eval_set=test \
        --class_label_dict_path=./data/voc12/cls_labels.npy \
        --train_list=./data/voc12/test.txt \
        --val_list=./data/voc12/val.txt \
        --infer_list=./data/voc12/test.txt \
        --cam_weights_name=./outputs/voc12/results/$MODEL/sess/cam.pth \
        --irn_weights_name=./outputs/voc12/results/$MODEL/sess/irn.pth \
        --cam_out_dir=./outputs/voc12/results/$MODEL/cam_test \
        --sem_seg_out_dir=./outputs/voc12/results/$MODEL/sem_seg_test \
        --ins_seg_out_dir=./outputs/voc12/results/$MODEL/ins_seg_test \
        --ir_label_out_dir=./outputs/voc12/results/$MODEL/irn_label_test \
        --cam_network=net.${MODEL}_cam \
        --irn_network=net.${MODEL}_irn \
        --log_name=./outputs/voc12/logs/${MODEL}_test \
        --make_sem_seg_pass=True \
        --eval_sem_seg_pass=True \
        --num_workers=64
done


