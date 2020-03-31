import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsl_survey.base.nets.networks import load_network
from wsl_survey.segmentation.irn.voc12 import dataloader


def create_features(file_list, voc12_root, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset = dataloader.VOC12ImageDataset(
        file_list,
        voc12_root=voc12_root,
        resize_long=(229, 229))
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=1,
                             pin_memory=True,
                             drop_last=True)

    model_ft, _ = load_network('resnet152', pretrained=True, finetune=False, image_size=299, return_backbone=True)
    model_ft.eval()
    features = []
    for pack in tqdm(data_loader, total=len(data_loader.dataset)):
        img = pack['img']
        img_name = pack['name'][0]
        y = model_ft(img)
        features.append({
            "img_name": img_name,
            "feature": y.cpu().detach().numpy()
        })
    np.save(output_file, features)


voc12_root = './datasets/voc2012/VOCdevkit/VOC2012/'
create_features("./data/voc12/train_aug.txt", voc12_root, './data/voc12/features/train_aug_features.npy')

create_features("./data/voc12/train.txt", voc12_root, './data/voc12/features/train_features.npy')

create_features("./data/voc12/val.txt", voc12_root, './data/voc12/features/val_features.npy')
