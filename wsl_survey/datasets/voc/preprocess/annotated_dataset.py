import os
import os.path
import os.path
from collections import OrderedDict

import xmltodict

from wsl_survey.datasets.utils import dict_to_file


def load_mapping(data_folder):
    filename = os.path.join(data_folder, 'class_mapping.csv')
    mapping = {}
    with open(filename, mode='r') as f:
        f.readline()
        for line in f.readlines():
            row = line.strip().split(',')
            mapping[row[1]] = row[0]
    return mapping


def load_annotations(data_folder):
    annotation_folder = os.path.join(data_folder, 'Annotations')
    annotation_files = os.listdir(annotation_folder)

    obj_list = []
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotation_folder, annotation_file)
        with open(annotation_path, mode='r') as f:
            xml = f.read()
        parsed_xml = xmltodict.parse(xml)
        filename = str(parsed_xml['annotation']['filename'])
        parsed_objs = parsed_xml['annotation']['object']
        if type(parsed_objs) == OrderedDict:
            parsed_objs = [parsed_objs]
        for parsed_obj in parsed_objs:
            obj = dict(parsed_obj['bndbox'])
            obj['class_name'] = parsed_obj['name']
            obj['image_filename'] = filename
            obj['id'] = filename.split('.')[0]
            obj_list.append(obj)
    return obj_list


def create_dataset(data_folder, labelled_folder):
    annotations = load_annotations(data_folder)
    mapping = load_mapping(labelled_folder)

    data = []
    for annotation in annotations:
        obj = annotation
        class_filename = annotation['class_name']
        obj['class_id'] = mapping[class_filename]
        data.append(obj)
    return data


item_extractor = lambda x: '%s,%s,%s,%s,%s,%s,%s,%s\n' % (x['id'], x[
    'class_id'], x['class_name'], x['xmin'], x['ymin'], x['xmax'], x['ymax'],
                                                          x['image_filename'])


def main(data_folder, labelled_folder, output_folder):
    annotations = create_dataset(data_folder, labelled_folder)
    header = 'id,class_id,class_name,xmin,ymin,xmax,ymax,image_filename\n'

    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.join(output_folder, 'annotations.csv')
    dict_to_file(annotations, filename, header, item_extractor)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create Annotated Dataset for Pascal VOC')
    parser.add_argument('--dataset_dir',
                        metavar='DIR',
                        help='path to data dir')
    parser.add_argument('--labelled_dir',
                        metavar='DIR',
                        help='path to labelled data dir')
    parser.add_argument('--output_dir',
                        metavar='DIR',
                        help='path to output dir')
    args = parser.parse_args()

    main(args.dataset_dir, args.labelled_dir, args.output_dir)
