import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

choices = []
paths = []
with open('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/train/test.txt', mode='r') as f:
    lines = f.readlines()
    for _ in range(12):
        value = random.choice(lines)
        value = value.strip() + '.jpg'
        path = os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars', value)
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

imgs_comb1 = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs[:6]))
imgs_comb1 = Image.fromarray(imgs_comb1)
imgs_comb2 = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs[6:]))
imgs_comb2 = Image.fromarray(imgs_comb2)

widths, heights = zip(*(i.size for i in [imgs_comb2, imgs_comb1]))
min_shape = sorted([(np.sum(i.size), i.size) for i in [imgs_comb2, imgs_comb1]])[0][1]

imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in [imgs_comb2, imgs_comb1]))
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save('car_examples.png', )
