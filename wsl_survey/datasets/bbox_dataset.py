import os
import os.path
import os.path

import torch.utils.data as data
from PIL import Image

from wsl_survey.datasets.utils import make_one_hot


class BBoxDataset(data.Dataset):
    def __init__(self,
                 dataset_folder,
                 image_folder,
                 split='train',
                 transform=None,
                 target_transform=None,
                 one_hot=False):
        self.image_folder = image_folder
        self.one_hot = one_hot
        self.images = []

        with open(os.path.join(dataset_folder, '%s.csv' % split),
                  mode='r') as f:
            header = f.readline().strip().split(',')
            for line in f.readlines():
                obj = dict(zip(header, line.strip().split(',')))
                self.images.append(
                    (obj['image_filename'], obj['class_id'],
                     (obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'])))

        self.transform = transform
        self.target_transform = target_transform

        self.classes = set([i[1] for i in self.images])

        print(
            '[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d'
            % (set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target, bbox = self.images[index]
        img = Image.open(os.path.join(self.image_folder, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.one_hot:
            target = make_one_hot(target, C=self.get_number_classes())
        return (img, path), target, bbox

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)


if __name__ == '__main__':
    train_dataset = BBoxDataset('datasets/voc2007/labelled',
                                'datasets/voc2007/data/JPEGImages',
                                split='train',
                                one_hot=True)
    for batch in train_dataset:
        print(batch)
