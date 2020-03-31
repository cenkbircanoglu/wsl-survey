import os
import shutil

import numpy as np

label_dict = np.load('./data/compcars/train/cls_labels_kmeans_431.npy', allow_pickle=True).item()
kmeans_results = 'data/compcars/train/kmeans_431'
for key, value in label_dict.items():
    key = key + '.jpg'
    label = np.argmax(value)
    path = os.path.join(kmeans_results, str(label), key.replace('/', '_').replace('data_image_', ''))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    src_path = os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars', key)
    shutil.copyfile(src_path, path)
