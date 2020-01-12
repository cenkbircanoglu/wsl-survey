import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2gray
from skimage.morphology import convex_hull_image
from skimage.morphology import disk
from skimage.morphology import erosion, opening, dilation, closing
from torch import multiprocessing, cuda
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsl_survey.segmentation.irn.misc import torchutils, imutils
from wsl_survey.segmentation.irn.voc12 import dataloader

cudnn.enabled = True
use_gpu = torch.cuda.is_available()


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')


def apply_morphology(img):
    img = np.pad(img, ((1, 0), (0, 0), (0, 0)), mode='constant')
    selem = disk(4)
    img = rgb2gray(img)
    eroded = erosion(img, selem)
    dilated = dilation(img, selem)
    opened = opening(img, selem)
    closed = closing(img, selem)
    hull1 = convex_hull_image(img == 0)

    return img, eroded, dilated, opened, closed, hull1


def _work_cpu(process_id, model, dataset, args):
    databin = dataset[process_id]

    data_loader = DataLoader(databin, shuffle=False,
                             num_workers=1,
                             pin_memory=False)

    with torch.no_grad():

        for iter, pack in tqdm(enumerate(data_loader), total=len(databin)):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0]) for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size,
                               mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [
                F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                              mode='bilinear', align_corners=False) for o in
                outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0,
                          :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            grayscaled, eroded, dilated, opened, closed, hull1 = apply_morphology(highres_cam.cpu().numpy())
            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(),
                     "high_res": highres_cam.cpu().numpy()})

            os.makedirs(os.path.join(args.cam_out_dir, 'grayscaled'), exist_ok=True)
            os.makedirs(os.path.join(args.cam_out_dir, 'eroded'), exist_ok=True)
            os.makedirs(os.path.join(args.cam_out_dir, 'dilated'), exist_ok=True)
            os.makedirs(os.path.join(args.cam_out_dir, 'opened'), exist_ok=True)
            os.makedirs(os.path.join(args.cam_out_dir, 'closed'), exist_ok=True)
            os.makedirs(os.path.join(args.cam_out_dir, 'hull1'), exist_ok=True)

            np.save(os.path.join(args.cam_out_dir, 'grayscaled', img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(),
                     "high_res": grayscaled})
            np.save(os.path.join(args.cam_out_dir, 'eroded', img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(),
                     "high_res": eroded})
            np.save(os.path.join(args.cam_out_dir, 'dilated', img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(),
                     "high_res": dilated})
            np.save(os.path.join(args.cam_out_dir, 'opened', img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(),
                     "high_res": opened})
            np.save(os.path.join(args.cam_out_dir, 'closed', img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(),
                     "high_res": closed})
            np.save(os.path.join(args.cam_out_dir, 'hull1', img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(),
                     "high_res": hull1})

            if process_id == args.num_workers - 1 and iter % (
                len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def _work_gpu(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False,
                             num_workers=args.num_workers // n_gpus,
                             pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in tqdm(enumerate(data_loader), total=len(databin)):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size,
                               mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [
                F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                              mode='bilinear', align_corners=False) for o in
                outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0,
                          :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(),
                     "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    if use_gpu:
        model.load_state_dict(torch.load(args.cam_weights_name + '.pth'),
                              strict=True)
    else:
        model.load_state_dict(torch.load(args.cam_weights_name + '.pth', map_location=torch.device('cpu')),
                              strict=True)
    model.eval()
    dataset = dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                       voc12_root=args.voc12_root,
                                                       scales=args.cam_scales,
                                                       class_label_dict_path=args.class_label_dict_path)
    print('[ ', end='')
    if use_gpu:
        n_gpus = torch.cuda.device_count()

        dataset = torchutils.split_dataset(dataset, n_gpus)
        multiprocessing.spawn(_work_gpu, nprocs=n_gpus,
                              args=(model, dataset, args), join=True)
    else:
        dataset = torchutils.split_dataset(dataset, args.num_workers)
        multiprocessing.spawn(_work_cpu, nprocs=args.num_workers,
                              args=(model, dataset, args),
                              join=True)
    print(']')

    torch.cuda.empty_cache()
