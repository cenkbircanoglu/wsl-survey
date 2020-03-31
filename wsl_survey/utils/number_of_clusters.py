import numpy as np
import pandas as pd

kmeans75_features = np.load('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/train/cls_labels_kmeans_16.npy',
                            allow_pickle=True).item()

with open('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/train/train.txt', mode='r') as f:
    tr_images = [x.strip() for x in f.readlines()]

with open('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/train/test.txt', mode='r') as f:
    te_images = [x.strip() for x in f.readlines()]

print(len(tr_images))
print(len(te_images))
d = {}
for key, value in kmeans75_features.items():
    label = np.argmax(value)
    d.setdefault(label, []).append(key)

data = []
for key, val in d.items():
    data.append({
        'label': key,
        'items': val,
        'cnt': len(val),
        'tr_cnt': len(set(tr_images).intersection(set(val))),
        'te_cnt': len(set(te_images).intersection(set(val)))
    })
print(d)
df = pd.DataFrame(data)
print(df.describe())
