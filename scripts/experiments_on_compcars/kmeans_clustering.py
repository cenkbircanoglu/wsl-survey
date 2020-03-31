import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def cluster_it(n_clusters):
    tr_features = np.load('./data/compcars/features/train_features.npy', allow_pickle=True)
    val_features = np.load('./data/compcars/features/test_features.npy', allow_pickle=True)
    features = tr_features.tolist() + val_features.tolist()

    df = pd.DataFrame.from_records(features)
    df.drop_duplicates('img_name', inplace=True)

    cluster = KMeans(n_clusters=n_clusters)

    df['feature'] = df['feature'].apply(lambda x: x[0].reshape(-1).tolist())
    X = np.array(df['feature'].values.tolist())
    cluster = cluster.fit(X)

    label_d = dict()
    category_size = len(set(cluster.labels_))
    for img_name, label in zip(df['img_name'].values, cluster.labels_):
        cluster_label = np.zeros(category_size)
        cluster_label[label] = 1
        label_d[img_name] = cluster_label

    np.save('./data/compcars/train/cls_labels_kmeans_%s.npy' % n_clusters, label_d)

    with open('./data/compcars/category_size.txt', mode='a') as f:
        f.write('%s %s\n' % ('kmeans_%s_id' % n_clusters, category_size))


if __name__ == '__main__':
    cluster_it(16)
    #cluster_it(431)
