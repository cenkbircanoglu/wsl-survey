import numpy as np
import pandas as pd
from sklearn import cluster

tr_aug_features = np.load('./data/voc12/features/train_aug_features.npy', allow_pickle=True)
tr_features = np.load('./data/voc12/features/train_features.npy', allow_pickle=True)
val_features = np.load('./data/voc12/features/val_features.npy', allow_pickle=True)
features = tr_aug_features.tolist() + tr_features.tolist() + val_features.tolist()

df = pd.DataFrame.from_records(features)
df.drop_duplicates('img_name', inplace=True)

cluster = cluster.KMeans(n_clusters=20)

df['feature'] = df['feature'].apply(lambda x: x[0].reshape(-1).tolist())
X = np.array(df['feature'].values.tolist())
cluster = cluster.fit(X)

label_d = dict()
category_size = len(set(cluster.labels_))
for img_name, label in zip(df['img_name'].values, cluster.labels_):
    cluster_label = np.zeros(category_size)
    cluster_label[label] = 1
    label_d[img_name] = cluster_label

np.save('./data/voc12/cls_kmeans_labels.npy', label_d)

with open('./data/voc12/category_size.txt', mode='a') as f:
    f.write('%s %s\n' % ('kmeans_id', category_size))
