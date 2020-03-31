import numpy as np
import pandas as pd
from scipy.spatial import distance
from torch.utils.data import DataLoader

from wsl_survey.base.nets.networks import load_network
from wsl_survey.segmentation.irn.voc12 import dataloader

dataset = dataloader.VOC12ImageDataset(
    "./data/voc12/val.txt",
    voc12_root='./datasets/voc2012/VOCdevkit/VOC2012/',
    resize_long=(229, 229))
data_loader = DataLoader(dataset,
                         batch_size=1,
                         shuffle=True,
                         num_workers=1,
                         pin_memory=True,
                         drop_last=True)

model_ft, _ = load_network('resnet152', pretrained=True, finetune=False, image_size=299, return_backbone=True)

features = np.load('./data/voc12/features/train_aug_features.npy', allow_pickle=True)

df = pd.DataFrame.from_records(features)
print(df.head())
results = []


def calc_distance(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    dist = distance.euclidean(x, y)
    return dist



for pack in data_loader:
    df1 = df.copy()
    img = pack['img']
    img_name = pack['name'][0]
    y = model_ft(img)
    y = y.cpu().detach().numpy()

    df1['dist'] = df1.apply(lambda x: calc_distance(x['feature'], y), axis=1)
    similar = df1[df1['dist'] == df1['dist'].min()]['img_name'].values[0]
    print(similar, img_name)
    results.append({'query': img_name, 'similar': similar})
    with open('./data/voc12/unique_class_mapping.txt', mode='a') as f:
        f.write('%s, %s\n' % (img_name, similar))
