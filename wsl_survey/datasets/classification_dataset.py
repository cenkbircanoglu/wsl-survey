import os
import os.path
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from wsl_survey.datasets.utils import make_one_hot


class ClassificationDataset(data.Dataset):
    def __init__(self,
                 dataset_folder,
                 image_folder,
                 split_type='train',
                 transform=None,
                 target_transform=None,
                 one_hot=False):
        self.image_folder = image_folder
        self.one_hot = one_hot
        self.images = []

        with open(os.path.join(dataset_folder, '%s.csv' % split_type),
                  mode='r') as f:
            header = f.readline().strip().split(',')
            for line in f.readlines():
                obj = dict(zip(header, line.strip().split(',')))
                self.images.append((obj['image_filename'], obj['class_id']))

        self.transform = transform
        self.target_transform = target_transform

        self.classes = set([i[1] for i in self.images])

        print(
            '[dataset] classification set=%s number of classes=%d  number of images=%d'
            % (split_type, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.image_folder, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.one_hot:
            target = make_one_hot(int(target), C=self.get_number_classes())
        else:
            target = int(target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return max(map(lambda x: int(x), self.classes)) + 1


def data_loader(args, split_type='train'):
    image_size = int(args.image_size)

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    tsfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])

    dataset = ClassificationDataset(args.dataset_dir,
                                    args.image_dir,
                                    split_type,
                                    transform=tsfm,
                                    one_hot=args.onehot)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset.get_number_classes(),
        shuffle=split_type == 'train',
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True)
    return loader


if __name__ == '__main__':
    train_dataset = ClassificationDataset('datasets/test/labelled',
                                          'datasets/test/images',
                                          split_type='train',
                                          one_hot=True)
    for img, target in train_dataset:
        print(target)
