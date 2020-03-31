import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances

tr_aug_features = np.load('./data/voc12/features/train_aug_features.npy', allow_pickle=True)
tr_features = np.load('./data/voc12/features/train_features.npy', allow_pickle=True)
val_features = np.load('./data/voc12/features/val_features.npy', allow_pickle=True)
features = tr_aug_features.tolist() + tr_features.tolist() + val_features.tolist()

df = pd.DataFrame.from_records(features)
df.drop_duplicates('img_name', inplace=True)
df['feature'] = df['feature'].apply(lambda x: x[0].reshape(-1).tolist())
X = np.array(df['feature'].values.tolist())
row_sums = X.sum(axis=1)
X = X / row_sums[:, np.newaxis]
print(X)
y = euclidean_distances(X)

print('mean: %s\t std: %s\t max: %s\t min: %s\n' % (np.mean(y), np.std(y), np.max(y), np.min(y)))
np.save('./data/voc12/features/euclidean_distance.npy', y)
