import os

import numpy as np
from torch.utils.data import DataLoader

from wsl_survey.base.nets.networks import load_network
from wsl_survey.segmentation.irn.voc12 import dataloader

train_dataset = dataloader.VOC12ImageDataset(
    "data/voc12/train_aug.txt",
    voc12_root='./datasets/voc2012/VOCdevkit/VOC2012/',
    resize_long=(229, 229))
train_data_loader = DataLoader(train_dataset,
                               batch_size=1,
                               shuffle=True,
                               num_workers=1,
                               pin_memory=True,
                               drop_last=True)

model_ft, _ = load_network('resnet152', pretrained=True, finetune=False, image_size=299, return_backbone=True)
model_ft.eval()
features = []
counter = 0
for pack in train_data_loader:
    img = pack['img']
    img_name = pack['name'][0]
    y = model_ft(img)
    features.append({
        "img_name": img_name,
        "feature": y.cpu().detach().numpy()
    })
    counter += 1
    print(counter)
np.save(os.path.join('training_features.npy'), features)
