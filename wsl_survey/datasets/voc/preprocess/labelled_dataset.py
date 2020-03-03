import os
import os.path
import os.path
import shutil

from wsl_survey.datasets.utils import generate_class_id, dict_to_file

object_categories = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def load_dataset_splits(data_folder):
    trainval_split = load_dataset_split(data_folder, split_type='trainval')
    tr_split = load_dataset_split(data_folder, split_type='train')
    val_split = load_dataset_split(data_folder, split_type='val')
    #test_split = load_dataset_split(data_folder, split_type='test')
    test_split = []

    return generate_class_id(trainval_split, tr_split, val_split, test_split)


def load_dataset_split(data_folder, split_type='train'):
    img_format = '%s.jpg'
    obj_list = []
    file_folder = os.path.join(data_folder, 'ImageSets', 'Main')
    for category in object_categories:
        file_path = os.path.join(file_folder,
                                 '%s_%s.txt' % (category, split_type))
        with open(file_path, mode='r') as f:
            for line in f.readlines():
                splitted = line.split()
                filename = str(splitted[0])
                if int(splitted[1]) == 1:
                    obj_list.append({
                        'class_name': category,
                        'id': filename,
                        'image_filename': img_format % filename,
                        'split_type': split_type
                    })
    return obj_list


item_extractor = lambda x: '%s,%s,%s,%s\n' % (x['id'], x['class_id'], x[
    'class_name'], x['image_filename'])
mapping_extractor = lambda x: '%s,%s\n' % (x['class_id'], x['class_name'])


def main(data_folder, output_folder):
    trainval_split, tr_set, val_set, test_set, class_mapping = load_dataset_splits(
        data_folder)
    header = 'id,class_id,class_name,image_filename\n'
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.join(output_folder, 'trainval.csv')
    dict_to_file(trainval_split, filename, header, item_extractor)

    filename = os.path.join(output_folder, 'train.csv')
    dict_to_file(tr_set, filename, header, item_extractor)

    filename = os.path.join(output_folder, 'val.csv')
    dict_to_file(val_set, filename, header, item_extractor)

    #filename = os.path.join(output_folder, 'test.csv')
    #dict_to_file(test_set, filename, header, item_extractor)

    header = 'class_id,class_name\n'
    filename = os.path.join(output_folder, 'class_mapping.csv')
    dict_to_file(class_mapping, filename, header, mapping_extractor)
    for filename in [ 'train.txt', 'trainval.txt', 'val.txt']:
        segmentation_file = os.path.join(data_folder, 'ImageSets',
                                         'Segmentation', filename)
        output_file = os.path.join(output_folder, filename)
        shutil.copy(segmentation_file, output_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create Labelled Dataset for Pascal VOC')
    parser.add_argument('--dataset_dir', type=str, help='path to data dir')
    parser.add_argument('--output_dir', type=str, help='path to output dir')
    args = parser.parse_args()

    main(args.dataset_dir, args.output_dir)
