import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

choices = []
paths = []

with open('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/arxiv_data/train_test_split/classification_test.txt', mode='r') as f:
    lines = f.readlines()
    for _ in range(96):
        value = random.choice(lines)
        value = value.strip()
        value = value.replace('data/image/', '').replace('/', '_')
        path = os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/results_1', value)
        im = plt.imread(path)
        paths.append(path)
        choices.append(im)

Image.MAX_IMAGE_PIXELS = 100000000  # For PIL Image error when handling very large images


def open_and_add_padding(i):
    old_im = Image.open(i)
    old_size = old_im.size

    new_size = (old_size[0] + 5, old_size[1] + 5)
    new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
    new_im.paste(old_im, (int((new_size[0] - old_size[0]) / 2),
                          int((new_size[1] - old_size[1]) / 2)))
    return new_im


imgs = [open_and_add_padding(i) for i in paths]
widths, heights = zip(*(i.size for i in imgs))
total_width = sum(widths)
max_height = max(heights)
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]

imgs_comb1 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs[:16]))
imgs_comb1 = Image.fromarray(imgs_comb1)

imgs_comb2 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs[16:32]))
imgs_comb2 = Image.fromarray(imgs_comb2)

imgs_comb3 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs[32:48]))
imgs_comb3 = Image.fromarray(imgs_comb3)

imgs_comb4 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs[48:64]))
imgs_comb4 = Image.fromarray(imgs_comb4)

imgs_comb5 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs[64:80]))
imgs_comb5 = Image.fromarray(imgs_comb5)

imgs_comb6 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs[80:96]))
imgs_comb6 = Image.fromarray(imgs_comb6)

arr = [imgs_comb1, imgs_comb2, imgs_comb3, imgs_comb4, imgs_comb5, imgs_comb6]
widths, heights = zip(*(i.size for i in arr))
min_shape = sorted([(np.sum(i.size), i.size) for i in arr])[0][1]

imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in arr))
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save('bbox_results_big.eps', format='eps')
