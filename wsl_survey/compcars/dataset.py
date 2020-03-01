import os
import os.path
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from wsl_survey.compcars.utils.parse_path import parse_path
from wsl_survey.datasets.utils import make_one_hot


def load_mapping(path):
    mapping = {}
    with open(path, mode='r') as f:
        for line in f.readlines():
            ind, uid = tuple(line.strip().split(','))
            mapping[int(uid)] = int(ind)
    return mapping


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
        make_mapping = load_mapping(dataset_folder + '_make_mapping.txt')
        model_mapping = load_mapping(dataset_folder + '_model_mapping.txt')
        year_mapping = load_mapping(dataset_folder + '_year_mapping.txt')
        with open(dataset_folder + '_%s.txt' % split_type, mode='r') as f:
            for line in f.readlines():
                make_uid, model_uid, year_uid, image_name, image_path = parse_path(line)
                make_id = make_mapping[make_uid]
                model_id = model_mapping[model_uid]
                year_id = year_mapping[year_uid]

                self.images.append((image_path, {'make_id': make_id, 'model_id': model_id, 'year': year_id}))

        self.transform = transform
        self.target_transform = target_transform

        self.classes = {
            'make_id': set([i[1]['make_id'] for i in self.images]),
            'model_id': set([i[1]['model_id'] for i in self.images]),
            'year': set([i[1]['year'] for i in self.images])
        }

        print(
            '[dataset] classification set=%s number of make classes=%d number of make model_classes=%d number of year_classes classes=%d  number of images=%d'
            % (split_type, len(self.classes['make_id']), len(self.classes['model_id']), len(self.classes['year']),
               len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.image_folder, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.one_hot:
            make_target = make_one_hot(int(target['make_id']), C=self.get_number_classes('make_id'))
            model_target = make_one_hot(int(target['model_id']), C=self.get_number_classes('model_id'))
            year_target = make_one_hot(int(target['year']), C=self.get_number_classes('year'))
        else:
            make_target = int(target['make_id'])
            model_target = int(target['model_id'])
            year_target = int(target['year'])
        if self.target_transform is not None:
            make_target = self.target_transform(make_target)
            model_target = self.target_transform(model_target)
            year_target = self.target_transform(year_target)

        return (img, path), {'make_id': make_target, 'model_id': model_target, 'year': year_target}

    def __len__(self):
        return len(self.images)

    def get_number_classes(self, category=None):
        if category:
            return max(map(lambda x: int(x), self.classes[category])) + 1
        else:
            return [self.get_number_classes('make_id'), self.get_number_classes('model_id'),
                    self.get_number_classes('year')]


def data_loader(args, split_type='train'):
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    tsfm = transforms.Compose([
        transforms.Resize((int(args.image_size), int(args.image_size))),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])

    dataset = ClassificationDataset(args.dataset_dir, args.image_dir, split_type, transform=tsfm, one_hot=args.onehot)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=split_type == 'train',
                                         num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return loader


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default=299, type=int)
    parser.add_argument("--onehot", default=True, type=bool)
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dataset_dir",
                        default='/Users/cenk.bircanoglu/workspace/icpr/arxiv_data/train_test_split/classification',
                        type=str)
    parser.add_argument("--image_dir", default='/Users/cenk.bircanoglu/workspace/icpr/', type=str)
    args = parser.parse_args()
    for i in data_loader(args, 'train'):
        print(i)
    # train_dataset = ClassificationDataset(
    #     '/Users/cenk.bircanoglu/workspace/icpr/arxiv_data/train_test_split/classification',
    #     '/Users/cenk.bircanoglu/workspace/icpr/',
    #     split_type='train',
    #     one_hot=True)
    # for img, target in train_dataset:
    #     print(target)
