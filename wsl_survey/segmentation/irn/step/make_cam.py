import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing, cuda
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsl_survey.segmentation.irn.misc import torchutils, imutils
from wsl_survey.segmentation.irn.voc12 import dataloader

cudnn.enabled = True
use_gpu = torch.cuda.is_available()


def _work_cpu(process_id, model, dataset, args):
    databin = dataset[process_id]

    data_loader = DataLoader(databin,
                             shuffle=False,
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

            strided_cam = torch.sum(
                torch.stack([
                    F.interpolate(torch.unsqueeze(o, 0),
                                  strided_size,
                                  mode='bilinear',
                                  align_corners=False)[0] for o in outputs
                ]), 0)

            highres_cam = [
                F.interpolate(torch.unsqueeze(o, 1),
                              strided_up_size,
                              mode='bilinear',
                              align_corners=False) for o in outputs
            ]
            highres_cam = torch.sum(torch.stack(highres_cam, 0),
                                    0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            # save cams
            np.save(
                os.path.join(args.cam_out_dir, img_name + '.npy'), {
                    "keys": valid_cat,
                    "cam": strided_cam.cpu(),
                    "high_res": highres_cam.cpu().numpy()
                })

            if process_id == args.num_workers - 1 and iter % (len(databin) //
                                                              4) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 4)), end='')


def _work_gpu(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin,
                             shuffle=False,
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

            outputs = [
                model(img[0].cuda(non_blocking=True)) for img in pack['img']
            ]

            strided_cam = torch.sum(
                torch.stack([
                    F.interpolate(torch.unsqueeze(o, 0),
                                  strided_size,
                                  mode='bilinear',
                                  align_corners=False)[0] for o in outputs
                ]), 0)

            highres_cam = [
                F.interpolate(torch.unsqueeze(o, 1),
                              strided_up_size,
                              mode='bilinear',
                              align_corners=False) for o in outputs
            ]
            highres_cam = torch.sum(torch.stack(highres_cam, 0),
                                    0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(
                os.path.join(args.cam_out_dir, img_name + '.npy'), {
                    "keys": valid_cat,
                    "cam": strided_cam.cpu(),
                    "high_res": highres_cam.cpu().numpy()
                })

            if process_id == n_gpus - 1 and iter % (len(databin) // 4) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 4)), end='')


def run(args):
    assert args.voc12_root is not None
    assert args.class_label_dict_path is not None
    assert args.train_list is not None
    assert args.cam_weights_name is not None
    assert args.cam_network is not None
    assert args.cam_out_dir is not None
    assert args.cam_network_module is not None

    dataset = dataloader.VOC12ClassificationDatasetMSF(
        args.train_list,
        voc12_root=args.voc12_root,
        scales=args.cam_scales,
        class_label_dict_path=args.class_label_dict_path)

    model = getattr(importlib.import_module(args.cam_network_module),
                    args.cam_network + 'CAM')()
    if use_gpu:
        model.load_state_dict(torch.load(args.cam_weights_name + '.pth'),
                              strict=True)
    else:
        model.load_state_dict(torch.load(args.cam_weights_name + '.pth',
                                         map_location=torch.device('cpu')),
                              strict=True)
    model.eval()

    print('[ ', end='')
    if use_gpu:
        n_gpus = torch.cuda.device_count()

        dataset = torchutils.split_dataset(dataset, n_gpus)
        multiprocessing.spawn(_work_gpu,
                              nprocs=n_gpus,
                              args=(model, dataset, args),
                              join=True)
    else:
        dataset = torchutils.split_dataset(dataset, args.num_workers)
        multiprocessing.spawn(_work_cpu,
                              nprocs=args.num_workers,
                              args=(model, dataset, args),
                              join=True)
    print(']')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./data/test1/VOC2012',
        class_label_dict_path='./data/voc12/cls_labels.npy',
        train_list='./data/test1/VOC2012/ImageSets/Segmentation/val.txt',
        cam_weights_name='./outputs/test1/results/resnet18/sess/cam.pth',
        cam_network='ResNet18',
        num_workers=1,
        cam_out_dir='./outputs/test1/results/resnet18/cam',
        cam_network_module='wsl_survey.segmentation.irn.net.resnet_cam',
    )
    args = parser.parse_args()
    os.makedirs(args.cam_out_dir, exist_ok=True)
    run(args)
