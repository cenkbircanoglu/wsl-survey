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
    parser.add_argument(
        "--out",
        default='./data/voc12/dog_train/',
        type=str)
    parser.add_argument(
        "--voc12_root",
        default="./datasets/voc2012/VOCdevkit/VOC2012",
        type=str)
    args = parser.parse_args()

    train_aug_list = dataloader.load_img_name_list(args.train_aug_list)
    train_list = dataloader.load_img_name_list(args.train_list)
    val_list = dataloader.load_img_name_list(args.val_list)

    train_val_name_list = np.concatenate([train_aug_list, train_list, val_list], axis=0)
    train_val_name_list = np.unique(train_val_name_list)
    cat_list = ['dog', 'train']
    cat_name_to_num = dict(zip(cat_list, range(len(cat_list))))
    label_list = dataloader.load_image_label_list_from_xml(
        train_val_name_list, args.voc12_root, cat_list=cat_list, cat_name_to_num=cat_name_to_num)

    total_label = np.zeros(2)

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
    np.save(os.path.join(args.out, 'cls_labels.npy'), d)
    with open(os.path.join(args.out, 'train_aug.txt'), mode='w')as f:
        for i in out_train_aug:
            f.write(i + '\n')
    with open(os.path.join(args.out, 'train.txt'), mode='w')as f:
        for i in out_train:
            f.write(i + '\n')
    with open(os.path.join(args.out, 'val.txt'), mode='w')as f:
        for i in out_val:
            f.write(i + '\n')
