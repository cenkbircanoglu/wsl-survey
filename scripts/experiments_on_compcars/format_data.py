import os

import numpy as np


def parse_path(x):
    x = x.strip()
    splitted = x.split('/')
    make_id = int(splitted[0])
    model_id = int(splitted[1])
    try:
        year = int(splitted[2])
    except ValueError:
        year = -1
    image_name = str(splitted[3])
    image_path = os.path.join('data/image', x)
    return make_id, model_id, year, image_name, image_path


if __name__ == '__main__':
    tr_images = []
    test_images = []
    make_set = set()
    model_set = set()
    year_set = set()
    make_year_set = set()
    model_year_set = set()
    with open('./data/compcars/arxiv_data/train_test_split/classification_train.txt', mode='r') as f:
        for line in f.readlines():
            make_uid, model_uid, year_uid, image_name, image_path = parse_path(line)
            make_year = '%s-%s' % (make_uid, year_uid)
            model_year = '%s-%s' % (model_uid, year_uid)
            make_set.add(make_uid)
            model_set.add(model_uid)
            year_set.add(year_uid)
            make_year_set.add(make_year)
            model_year_set.add(model_year)

            tr_images.append(
                {
                    'image_path': image_path,
                    'make_id': make_uid,
                    'model_id': model_uid,
                    'make_year': make_year,
                    'model_year': model_year,
                    'year': year_uid,
                    'img_name': image_path.replace('.jpg', '')
                }
            )
    with open('./data/compcars/arxiv_data/train_test_split/classification_test.txt', mode='r') as f:
        for line in f.readlines():
            make_uid, model_uid, year_uid, image_name, image_path = parse_path(line)
            make_year = '%s-%s' % (make_uid, year_uid)
            model_year = '%s-%s' % (model_uid, year_uid)
            make_set.add(make_uid)
            model_set.add(model_uid)
            year_set.add(year_uid)
            make_year_set.add(make_year)
            model_year_set.add(model_year)

            test_images.append(
                {
                    'image_path': image_path,
                    'make_id': make_uid,
                    'model_id': model_uid,
                    'make_year': make_year,
                    'model_year': model_year,
                    'year': year_uid,
                    'img_name': image_path.replace('.jpg', '')
                }
            )

    make_mapping = {uid: i for i, uid in enumerate(make_set)}
    model_mapping = {uid: i for i, uid in enumerate(model_set)}
    year_mapping = {uid: i for i, uid in enumerate(year_set)}
    make_year_mapping = {uid: i for i, uid in enumerate(make_year_set)}
    model_year_mapping = {uid: i for i, uid in enumerate(model_year_set)}

    with open('./data/compcars/train/train.txt', mode='w') as f:
        for img in tr_images:
            f.write('%s\n' % img['img_name'])

    with open('./data/compcars/train/test.txt', mode='w') as f:
        for img in test_images:
            f.write('%s\n' % img['img_name'])

    make_d = dict()
    model_d = dict()
    year_d = dict()
    make_year_d = dict()
    model_year_d = dict()

    for image in tr_images + test_images:
        img_name = image['img_name']

        make_label = np.zeros(len(make_set))
        make_label[make_mapping[image['make_id']]] = 1
        make_d[img_name] = make_label

        model_label = np.zeros(len(model_set))
        model_label[model_mapping[image['model_id']]] = 1
        model_d[img_name] = model_label

        year_label = np.zeros(len(year_set))
        year_label[year_mapping[image['year']]] = 1
        year_d[img_name] = year_label

        make_year_label = np.zeros(len(make_year_mapping))
        make_year_label[make_year_mapping[image['make_year']]] = 1
        make_year_d[img_name] = make_year_label

        model_year_label = np.zeros(len(model_year_set))
        model_year_label[model_year_mapping[image['model_year']]] = 1
        model_year_d[img_name] = model_year_label

    np.save('./data/compcars/train/cls_labels_make.npy', make_d)
    np.save('./data/compcars/train/cls_labels_model.npy', model_d)
    np.save('./data/compcars/train/cls_labels_year.npy', year_d)
    np.save('./data/compcars/train/cls_labels_make_year.npy', make_year_d)
    np.save('./data/compcars/train/cls_labels_model_year.npy', model_year_d)

    with open('./data/compcars/train/category_size.txt', mode='w') as f:
        f.write('%s %s\n' % ('make_set', len(make_set)))
        f.write('%s %s\n' % ('model_set', len(model_set)))
        f.write('%s %s\n' % ('year_set', len(year_set)))
        f.write('%s %s\n' % ('make_year_label', len(make_year_set)))
        f.write('%s %s\n' % ('model_year_set', len(model_year_set)))
