from random import randrange

import numpy as np

if __name__ == '__main__':
    make_d = np.load('./data/compcars/train/cls_labels_make.npy', allow_pickle=True)
    random_d = {}

    for i in make_d.item().items():
        label = np.zeros(100)
        label[randrange(100)] = 1
        random_d[i[0]] = label

    np.save('./data/compcars/train/cls_labels_random_100.npy', random_d)
