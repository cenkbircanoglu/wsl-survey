import argparse

import numpy as np

from wsl_survey.segmentation.irn.voc12 import dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='./data/voc12/train_aug.txt', type=str)
    parser.add_argument("--out", default='./data/voc12/cls_unique_labels.npy', type=str)
    args = parser.parse_args()

    dataset_list = dataloader.load_img_name_list(args.dataset_path)

    d = dict()
    for i, img_name in enumerate(dataset_list):
        label = np.zeros(len(dataset_list))
        label[i] = 1
        d[img_name] = label
    np.save(args.out, d)
    with open('./data/voc12/category_size.txt', mode='a') as f:
        f.write('%s %s\n' % ('category id', 20))
        f.write('%s %s\n' % ('unique_ids', len(dataset_list)))
