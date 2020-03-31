import os

import numpy as np

paths = './datasets/DISFA/DISFA/ActionUnit_Labels/'
root_paths = os.listdir(paths)
d = {}
label_set = set()
for root_path in root_paths:
    subpath = os.path.join(paths, root_path)
    if '.DS_Store' not in subpath:

        for path in os.listdir(subpath):
            file_path = os.path.join(subpath, path)
            if '.DS_Store' not in file_path:
                with open(file_path, mode='r') as f:
                    for line in f.readlines():
                        num, conf = line.strip().split(',')
                        if int(conf):
                            label = path.split('_')[1].split('.')[0] + '_' + conf
                            d.setdefault('%s/%s' % (path.split('_')[0], num), []).append(label)
                            label_set.add(label)

labels = {}
for i, label in enumerate(label_set):
    labels[label] = i

dictionary = dict()
for key, values in d.items():
    total_label = np.zeros(len(label_set))
    for label in values:
        total_label[labels[label]] = 1
    dictionary[key] = total_label

print(dictionary)
print(len(label_set))
print(len(labels))
os.makedirs('./data/disfa/', exist_ok=True)
np.save('./data/disfa/cls_labels_conf.npy', dictionary)
