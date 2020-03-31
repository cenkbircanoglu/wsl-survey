import os
import random

import numpy as np
from PIL import Image

paths = []
with open('/Users/cenk.bircanoglu/wsl/wsl_survey/data/compcars/train/test.txt', mode='r') as f:
    lines = f.readlines()
    for _ in range(8):
        value = random.choice(lines)
        value = value.strip() + '.png'
        paths.append(value)

Image.MAX_IMAGE_PIXELS = 100000000  # For PIL Image error when handling very large images


def open_and_add_padding(i):
    old_im = Image.open(i)
    old_size = old_im.size

    new_size = (old_size[0] + 5, old_size[1] + 5)
    new_im = Image.new("RGB", new_size, color=128)  ## luckily, this is already black!
    new_im.paste(old_im, (int((new_size[0] - old_size[0]) / 2),
                          int((new_size[1] - old_size[1]) / 2)))
    return new_im


paths1 = [
    os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/make/results/resnet50/irn_label', i)
    for i in paths]
paths2 = [
    os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/model/results/resnet50/irn_label', i)
    for i in paths]
paths3 = [
    os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/year/results/resnet50/irn_label', i)
    for i in paths]
paths4 = [
    os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/make_year/results/resnet50/irn_label',
                 i) for i in paths]
paths5 = [
    os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/make_year/results/resnet50/irn_label',
                 i) for i in paths]
paths6 = [
    os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/kmeans_75/results/resnet50/irn_label',
                 i) for i in paths]
paths7 = [
    os.path.join('/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/kmeans_75/results/resnet50/irn_label',
                 i) for i in paths]
paths8 = [os.path.join(
    '/Users/cenk.bircanoglu/wsl/wsl_survey/compcars_outputs/compcars/kmeans_431/results/resnet50/irn_label', i) for i in
          paths]

imgs1 = [open_and_add_padding(i) for i in paths1]
imgs2 = [open_and_add_padding(i) for i in paths2]
imgs3 = [open_and_add_padding(i) for i in paths3]
imgs4 = [open_and_add_padding(i) for i in paths4]
imgs5 = [open_and_add_padding(i) for i in paths5]
imgs6 = [open_and_add_padding(i) for i in paths6]
imgs7 = [open_and_add_padding(i) for i in paths7]
imgs8 = [open_and_add_padding(i) for i in paths8]

widths, heights = zip(*(i.size for i in imgs1))
total_width = sum(widths)
max_height = max(heights)
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted([(np.sum(i.size), i.size) for i in imgs1])[0][1]

imgs_comb1 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs1))
imgs_comb1 = Image.fromarray(imgs_comb1)

imgs_comb2 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs2))
imgs_comb2 = Image.fromarray(imgs_comb2)

imgs_comb3 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs3))
imgs_comb3 = Image.fromarray(imgs_comb3)

imgs_comb4 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs4))
imgs_comb4 = Image.fromarray(imgs_comb4)

imgs_comb5 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs5))
imgs_comb5 = Image.fromarray(imgs_comb5)

imgs_comb6 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs6))
imgs_comb6 = Image.fromarray(imgs_comb6)

imgs_comb7 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs7))
imgs_comb7 = Image.fromarray(imgs_comb7)

imgs_comb8 = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs8))
imgs_comb8 = Image.fromarray(imgs_comb8)

l = [imgs_comb1, imgs_comb2, imgs_comb3, imgs_comb4, imgs_comb5, imgs_comb6, imgs_comb7, imgs_comb8]
widths, heights = zip(*(i.size for i in l))
min_shape = sorted([(np.sum(i.size), i.size) for i in l])[0][1]

imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in l))
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save('cam_results.png', )
