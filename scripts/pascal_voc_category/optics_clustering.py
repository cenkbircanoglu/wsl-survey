import numpy as np
import pandas as pd
from sklearn import cluster, metrics

tr_aug_features = np.load('./data/voc12/features/train_aug_features.npy', allow_pickle=True)
tr_features = np.load('./data/voc12/features/train_features.npy', allow_pickle=True)
val_features = np.load('./data/voc12/features/val_features.npy', allow_pickle=True)
features = tr_aug_features.tolist() + tr_features.tolist() + val_features.tolist()

df = pd.DataFrame.from_records(features)
df.drop_duplicates('img_name', inplace=True)
df['feature'] = df['feature'].apply(lambda x: x[0].reshape(-1).tolist())
X = np.array(df['feature'].values.tolist())

for eps in range(2, 50, 3):
    for min_sample in [2, 3, 4, 8, 16, 32]:
        print(eps, min_sample)
        cls = cluster.OPTICS(eps=eps, n_jobs=-1, min_samples=min_sample)
        cls = cls.fit(X)
        labels = cls.labels_
        label_d = dict()
        category_size = len(set(cls.labels_))
        for img_name, label in zip(df['img_name'].values, cls.labels_):
            cluster_label = np.zeros(category_size)
            cluster_label[label] = 1
            label_d[img_name] = cluster_label
        print(eps, category_size, min_sample)
        if len(set(cls.labels_)) > 1:
            print("Silhouette Coefficient: %0.3f"
                  % metrics.silhouette_score(X, labels))

        np.save('./data/voc12/cls_optics%s_%s_labels.npy' % (eps, min_sample), label_d)

        with open('./data/voc12/category_size.txt', mode='a') as f:
            f.write('%s=%s %s %s\n' % ('optics_eps', eps, min_sample, category_size))
