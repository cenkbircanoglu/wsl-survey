import argparse

import numpy as np

from wsl_survey.segmentation.irn_unique.voc12 import dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_aug_list",
        default=
        '/Users/cenk.bircanoglu/wsl/wsl_survey/data/voc12/train_aug.txt',
        type=str)
    parser.add_argument(
        "--train_list",
        default='/Users/cenk.bircanoglu/wsl/wsl_survey/data/voc12/train.txt',
        type=str)
    parser.add_argument(
        "--val_list",
        default='/Users/cenk.bircanoglu/wsl/wsl_survey/data/voc12/val.txt',
        type=str)
    parser.add_argument(
        "--test_list",
        default='/Users/cenk.bircanoglu/wsl/wsl_survey/data/voc12/test.txt',
        type=str)
    parser.add_argument(
        "--out",
        default=
        "/Users/cenk.bircanoglu/wsl/wsl_survey/data/voc12/cls_labels_test.npy",
        type=str)
    parser.add_argument(
        "--voc12_root",
        default=
        "/Users/cenk.bircanoglu/wsl/wsl_survey/datasets/voc2012/tmp/VOCdevkit/VOC2012",
        type=str)
    args = parser.parse_args()

    test_list = dataloader.load_img_name_list(args.test_list)

    train_val_name_list = np.concatenate([test_list], axis=0)
    train_val_name_list = np.unique(train_val_name_list)

    label_list = dataloader.load_image_label_list_from_xml(
        train_val_name_list, args.voc12_root)

    total_label = np.zeros(20)

    d = dict()
    for img_name, label in zip(train_val_name_list, label_list):
        d[img_name] = label
        total_label += label

    print(total_label)
    np.save(args.out, d)
