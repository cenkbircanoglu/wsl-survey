import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from wsl_survey.base.nets.networks import load_network
from wsl_survey.segmentation.irn.voc12 import dataloader

train_dataset = dataloader.VOC12ImageDataset(
    "data/voc12/val.txt",
    voc12_root='./datasets/voc2012/VOCdevkit/VOC2012/',
    resize_long=(229, 229))
train_data_loader = DataLoader(train_dataset,
                               batch_size=1,
                               shuffle=True,
                               num_workers=1,
                               pin_memory=True,
                               drop_last=False)

model_ft, _ = load_network('resnet152', pretrained=True, finetune=False, image_size=299, return_backbone=True)

tr_dir = 'features/pascal_voc12/training'

features = np.load('training_features.npy', allow_pickle=True)

df = pd.DataFrame.from_records(features)
print(df.head())
results = []
for pack in train_data_loader:
    df1 = df.copy()
    img = pack['img']
    img_name = pack['name'][0]
    y = model_ft(img)
    y = y.cpu().detach().numpy()
    df1['dist'] = df1['feature'].apply(lambda x: np.linalg.norm(x - y))
    print(df1[df1['dist'] == df1['dist'].min()]['img_name'].values[0], img_name)
    results.append({'query': img_name, 'similar': df1[df1['dist'] == df1['dist'].min()]['img_name'].values[0]})

dataset = dataloader.VOC12ClassificationDatasetMSF(
    "data/voc12/train_aug.txt",
    voc12_root='./datasets/voc2012/VOCdevkit/VOC2012/',
    class_label_dict_path='./data/voc12/cls_labels_unique.npy')

data_loader = DataLoader(dataset,
                         batch_size=1,
                         shuffle=True,
                         num_workers=1,
                         pin_memory=True,
                         drop_last=False)

name_label_mapping = {}
for pack in data_loader:
    img_name = pack['name'][0]
    label = pack['label'][0]
    name_label_mapping[img_name] = label

labelled_results = []
with open('classes.txt', mode='w') as f:
    for res in results:
        label = name_label_mapping[res['similar']]
        f.write('%s, %s, %s' % (res['query'], res['similar'], label))
