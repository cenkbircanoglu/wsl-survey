from collections import Counter

import numpy as np

d = np.load('./data/voc12/cls_optics_labels.npy', allow_pickle=True).item()
counter = Counter()
for key, val in d.items():
    counter.update({int(np.argmax(val)): 1})

print(counter)
