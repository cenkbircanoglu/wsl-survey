import json
import os
import os.path

from wsl_survey.datasets.utils import generate_class_id, dict_to_file


def load_json_dataset(data_folder, split_type='train'):
    version = '2017' if '2017' in data_folder else '2014'
    filename = os.path.join(data_folder,
                            'instances_%s%s.json' % (split_type, version))

    with open(filename, mode='r') as json_file:
        data = json.load(json_file)
    annotations = data['annotations']
    categories = {}
    images = {}
    for category in data['categories']:
        categories[category['id']] = category

    for image in data['images']:
        images[image['id']] = image
    items = []
    for annotation in annotations:
        img_id = annotation['image_id']
        category_id = annotation['category_id']
        category = categories[category_id]
        img = images[img_id]
        bbox = annotation['bbox']
        obj = {
            'category_id': category_id,
            'id': str(annotation['id']),
            'class_name': category['name'],
            'image_filename': img['file_name'],
            'parent_category': category['supercategory'],
            'xmin': bbox[0],
            'ymin': bbox[1],
            'xmax': bbox[2],
            'ymax': bbox[3]
        }
        items.append(obj)
    return items


item_extractor = lambda x: '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (x['id'], x[
    'class_id'], x['class_name'], x['image_filename'], x['xmin'], x['ymin'], x[
        'xmax'], x['ymax'], x['category_id'], x['parent_category'])
mapping_extractor = lambda x: '%s,%s\n' % (x['class_id'], x['class_name'])


def main(data_folder, output_folder):
    tr_set = load_json_dataset(data_folder, 'train')
    val_set = load_json_dataset(data_folder, 'val')
    _, tr_set, val_set, _, class_mapping = generate_class_id([], tr_set,
                                                             val_set, [])

    header = 'id,class_id,class_name,xmin,ymin,xmax,ymax,image_filename,category_id,parent_category\n'
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.join(output_folder, 'train.csv')
    dict_to_file(tr_set, filename, header, item_extractor)

    filename = os.path.join(output_folder, 'val.csv')
    dict_to_file(val_set, filename, header, item_extractor)

    header = 'class_id,class_name\n'
    filename = os.path.join(output_folder, 'class_mapping.csv')
    dict_to_file(class_mapping, filename, header, mapping_extractor)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create Labelled Dataset for Coco 2017')
    parser.add_argument('--dataset_dir',
                        metavar='DIR',
                        help='path to data dir')
    parser.add_argument('--output_dir',
                        metavar='DIR',
                        help='path to output dir')
    args = parser.parse_args()

    main(args.dataset_dir, args.output_dir)
