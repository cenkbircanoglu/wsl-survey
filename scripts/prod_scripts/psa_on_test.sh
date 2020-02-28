################# Train a classification network to get CAMs. #################
python3 wsl_survey/segmentation/psa/train_cls.py \
    --lr=0.1 \
    --batch_size=16 \
    --max_epoches=15 \
    --crop_size=448 \
    --network=network.resnet38_cls \
    --voc12_root=./datasets/voc2012/VOCdevkit/VOC2012 \
    --weights=res38_cls.pth \
    --wt_dec=5e-4 \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/train.txt \
    --class_label_dict_path=./data/voc12/cls_labels.npy


################# Generate labels for AffinityNet by applying dCRF on CAMs. #################
python3 wsl_survey/segmentation/psa/infer_aff.py \
    --infer_list=./data/voc12/train_aug.txt \
    --voc12_root=./datasets/voc2012/VOCdevkit/VOC2012 \
    --network=network.resnet38_cls \
    --weights=res38_cls.pth \
    --out_cam=./outputs/test/results/out_cam \
    --out_la_crf=./outputs/test/results/out_la_crf \
    --out_ha_crf=./outputs/test/results/out_ha_crf \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/train.txt \
    --class_label_dict_path=./data/voc12/cls_labels.npy

################# (Optional) Check the accuracy of CAMs. #################

python3 wsl_survey/segmentation/psa/infer_cls.py \
    --infer_list=./data/voc12/val.txt \
    --voc12_root=./datasets/voc2012/VOCdevkit/VOC2012 \
    --network=network.resnet38_cls \
    --weights=res38_cls.pth \
    --out_cam_pred=./outputs/test/results/out_cam_pred \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/train.txt \
    --class_label_dict_path=./data/voc12/cls_labels.npy


################# Train AffinityNet with the labels #################

python3 wsl_survey/segmentation/psa/train_aff.py \
    --lr=0.1 \
    --batch_size=8 \
    --max_epoches=8 \
    --crop_size=448 \
    --voc12_root=./datasets/voc2012/VOCdevkit/VOC2012 \
    --network=network.resnet38_aff \
    --weights=res38_cls.pth \
    --wt_dec=5e-4 \
    --la_crf_dir=./outputs/test/results/la_crf_dir \
    --ha_crf_dir=./outputs/test/results/ha_crf_dir \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/train.txt \
    --class_label_dict_path=./data/voc12/cls_labels.npy

################# 4. Perform Random Walks on CAMs #################

python3 wsl_survey/segmentation/psa/infer_aff.py \
    --infer_list=./data/voc12/train.txt \
    --voc12_root=./datasets/voc2012/VOCdevkit/VOC2012 \
    --network=network.resnet38_aff \
    --weights=res38_cls.pth \
    --cam_dir=./outputs/test/results/cam_dir \
    --out_rw=./outputs/test/results/out_rw \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/train.txt \
    --class_label_dict_path=./data/voc12/cls_labels.npy


python3 wsl_survey/segmentation/psa/infer_aff.py \
    --infer_list=./data/voc12/val.txt \
    --voc12_root=./datasets/voc2012/VOCdevkit/VOC2012 \
    --network=network.resnet38_aff \
    --weights=res38_cls.pth \
    --cam_dir=./outputs/test/results/cam_dir \
    --out_rw=./outputs/test/results/out_rw \
    --train_list=./data/voc12/train_aug.txt \
    --val_list=./data/voc12/train.txt \
    --class_label_dict_path=./data/voc12/cls_labels.npy
