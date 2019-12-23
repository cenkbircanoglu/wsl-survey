import os
import shutil

import pandas as pd
from PIL import Image, ImageOps


def get_img_files(sources):
    img_files = set()
    for source in sources:
        df = pd.read_csv(source)
        filenames = df['image_filename'].to_list()
        for filename in filenames:
            img_files.add(filename)
    return img_files


def _resize_to_square(img, fill=0):
    if img.width > img.height:
        border = (0, (img.width - img.height) // 2)
    elif img.width < img.height:
        border = ((img.height - img.width) // 2, 0)
    else:
        return img.copy()
    return ImageOps.expand(img, border, fill)


def take_n_from_each_group(source, target, group_key='class_id', n=10):
    output_folder = os.path.dirname(target)
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(source, dtype={'id': str})
    df = df.groupby(group_key).head(n)
    df.to_csv(target, index=False)


sources = [
    'datasets/voc2007/labelled/trainval.csv',
    'datasets/voc2007/labelled/train.csv', 'datasets/voc2007/labelled/val.csv',
    'datasets/voc2007/labelled/test.csv',
    'datasets/voc2007/annotated/annotations.csv',
    'datasets/voc2007/annotated/annotations.csv',
    'datasets/voc2007/annotated/annotations.csv',
    'datasets/voc2007/annotated/annotations.csv',
    'datasets/voc2007/annotated/annotations.csv'
]
targets = [
    'datasets/test/labelled/trainval.csv', 'datasets/test/labelled/train.csv',
    'datasets/test/labelled/val.csv', 'datasets/test/labelled/test.csv',
    'datasets/test/annotated/annotations.csv',
    'datasets/test/annotated/trainval.csv',
    'datasets/test/annotated/train.csv', 'datasets/test/annotated/val.csv',
    'datasets/test/annotated/test.csv'
]
for source, target in zip(sources, targets):
    take_n_from_each_group(source, target)

shutil.copy('datasets/voc2007/labelled/class_mapping.csv',
            'datasets/test/labelled/class_mapping.csv')

img_files = get_img_files(targets)
source = 'datasets/voc2007/data/JPEGImages'
target = 'datasets/test/images'

shutil.rmtree(target)
os.makedirs(target, exist_ok=True)

for img_file in img_files:
    im = Image.open(os.path.join(source, img_file))
    size = (256, 256)
    im = ImageOps.fit(_resize_to_square(im), size, Image.ANTIALIAS)
    im.save(os.path.join(target, img_file), "JPEG")

source = 'datasets/voc2007/VOCdevkit/VOC2007/Annotations'
target = 'datasets/test/voc_root/Annotations'

os.makedirs(target, exist_ok=True)
for img_file in img_files:
    xml_file = img_file.replace('.jpg', '.xml')
    shutil.copy(os.path.join(source, xml_file), os.path.join(target, xml_file))

source = 'datasets/voc2007/VOCdevkit/VOC2007/SegmentationClass'
target = 'datasets/test/voc_root/SegmentationClass'

os.makedirs(target, exist_ok=True)
for img_file in img_files:
    png_file = img_file.replace('.jpg', '.png')
    shutil.copy(os.path.join(source, png_file), os.path.join(target, png_file))

source = 'datasets/voc2007/VOCdevkit/VOC2007/SegmentationObject'
target = 'datasets/test/voc_root/SegmentationObject'

os.makedirs(target, exist_ok=True)
for img_file in img_files:
    png_file = img_file.replace('.jpg', '.png')
    shutil.copy(os.path.join(source, png_file), os.path.join(target, png_file))
