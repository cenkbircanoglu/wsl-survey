import argparse
import os

import numpy as np

from wsl_survey.segmentation.irn.voc12 import dataloader
from wsl_survey.segmentation.irn.voc12.dataloader import decode_int_filename

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_aug_list",
        default='./data/voc12/train_aug.txt',
        type=str)
    parser.add_argument(
        "--train_list",
        default='./data/voc12/train.txt',
        type=str)
    parser.add_argument(
        "--val_list",
        default='./data/voc12/val.txt',
        type=str)
    parser.add_argument(
        "--test_list",
        default='./data/test1/VOC2012/ImageSets/Segmentation/test.txt',
        type=str)
    parser.add_argument("--out", default='./data/voc12/', type=str)
    parser.add_argument("--subset", default=None, type=str)
    parser.add_argument(
        "--voc12_root",
        default="./datasets/voc2012/VOCdevkit/VOC2012",
        type=str)
    args = parser.parse_args()
    out_dir = os.path.join(args.out, args.subset)
    os.makedirs(out_dir, exist_ok=True)
    train_aug_list = dataloader.load_img_name_list(args.train_aug_list)
    train_list = dataloader.load_img_name_list(args.train_list)
    val_list = dataloader.load_img_name_list(args.val_list)

    train_val_name_list = np.concatenate(
        [train_aug_list, train_list, val_list], axis=0)
    train_val_name_list = np.unique(train_val_name_list)
    cat_list = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    subsets = {
        'subset1': ['cat', 'dog'],
        'subset2': ['bus', 'car'],
        'subset3': ['cat', 'dog', 'horse'],
        'subset4': ['bus', 'car', 'train'],
        'subset5': ['cat', 'dog', 'bus'],
        'subset6': ['cat', 'dog', 'horse', 'bus'],
        'subset7': ['cat', 'bus', 'car', 'train'],
        'subset8': ['cat', 'dog', 'horse', 'bus', 'car'],
        'subset9': ['cat', 'dog', 'bus', 'car', 'train'],
        'subset10': ['cat', 'dog', 'horse', 'bus', 'car', 'train']
    }
    if subsets:
        cat_list = subsets.get(args.subset, cat_list)
    print(cat_list)
    label_list = dataloader.load_image_label_list_from_xml(
        train_val_name_list, args.voc12_root, cat_list=cat_list)

    total_label = np.zeros(20)

    d = dict()
    out_train_aug = []
    out_train = []
    out_val = []
    for img_name, label in zip(train_val_name_list, label_list):
        if np.any(label):
            d[img_name] = label
            total_label += label
            if img_name in train_aug_list:
                out_train_aug.append(decode_int_filename(img_name))
            if img_name in train_list:
                out_train.append(decode_int_filename(img_name))
            if img_name in val_list:
                out_val.append(decode_int_filename(img_name))

    print(total_label)
    np.save(os.path.join(out_dir, 'cls_labels.npy'), d)
    with open(os.path.join(out_dir, 'train_aug.txt'), mode='w')as f:
        for i in out_train_aug:
            f.write(i + '\n')
    with open(os.path.join(out_dir, 'train.txt'), mode='w')as f:
        for i in out_train:
            f.write(i + '\n')
    with open(os.path.join(out_dir, 'val.txt'), mode='w')as f:
        for i in out_val:
            f.write(i + '\n')
