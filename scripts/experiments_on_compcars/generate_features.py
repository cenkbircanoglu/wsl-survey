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
                             drop_last=False)

    model_ft, _ = load_network('resnet152', pretrained=True, finetune=False, image_size=299, return_backbone=True)
    model_ft.eval()
    model_ft.cuda()
    features = []
    for pack in tqdm(data_loader, total=len(data_loader.dataset)):
        img = pack['img'].cuda()
        img_name = pack['name'][0]
        y = model_ft(img)
        features.append({
            "img_name": img_name,
            "feature": y.cpu().detach().numpy()
        })
    np.save(output_file, features)


voc12_root = './data/compcars/'
create_features("./data/compcars/train/train.txt",
                voc12_root, './data/compcars/features/train_features.npy')

create_features("./data/compcars/train/test.txt",
                voc12_root, './data/compcars/features/test_features.npy')

