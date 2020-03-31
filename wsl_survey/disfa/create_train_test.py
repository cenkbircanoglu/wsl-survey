import random

import numpy as np

d = np.load('./data/disfa/cls_labels_conf.npy', allow_pickle=True).item()

train = []
test = []
for i in d.keys():
    if random.random() > 0.2:
        train.append(i.replace('.jpg', ''))
    else:
        test.append(i.replace('.jpg', ''))
print(len(train))
print(len(test))

with open('./data/disfa/conf_train.txt', mode='w') as f:
    for i in train:
        f.write('%s\n' % i)

with open('./data/disfa/conf_test.txt', mode='w') as f:
    for i in test:
        f.write('%s\n' % i)
