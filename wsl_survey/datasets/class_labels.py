import os

import numpy as np


def load_csv(data_folder, split_type='train'):
    filename = os.path.join(data_folder, '%s.csv' % split_type)
    if not os.path.exists(filename):
        return []
    items = []
    with open(filename, mode='r') as f:
        header = f.readline().strip().split(',')
        for line in f.readlines():
            row = line.strip().split(',')
            item = dict(zip(header, row))
            items.append({'id': item['id'], 'class_id': item['class_id']})

    return items


def one_hot_mappings(items):
    labels = set([i['class_id'] for i in items])

    d = dict()
    for item in items:
        total_label = np.zeros(len(labels))
        img_name = item['id']
        label = item['class_id']
        total_label[int(label)] = 1
        d[img_name] = total_label

    return d


def create_class_labels(source, target):
    filename = os.path.join(target, 'cls_labels')
    trainval = load_csv(source, 'trainval')
    train = load_csv(source, 'train')
    val = load_csv(source, 'val')
    #test = load_csv(source, 'test')
    items = trainval + train + val #+ test
    mapping = one_hot_mappings(items)
    np.save(filename, mapping)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create Class Labels')
    parser.add_argument('--source_dir', metavar='DIR', help='path to data dir')
    parser.add_argument('--target_dir',
                        metavar='DIR',
                        help='path to labelled data dir')
    args = parser.parse_args()

    create_class_labels(args.source_dir, args.target_dir)
