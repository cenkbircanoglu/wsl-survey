import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

from wsl_survey.irn import dataloader, imutils
from wsl_survey.irn.net.resnet50_cam import CAM

use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.enabled = True


def _work(model, dataset, args, cam_out_dir):
    data_loader = DataLoader(dataset,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=False)

    with torch.no_grad():
        if use_cuda:
            model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            if use_cuda:
                label = label.cuda()

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            if use_cuda:
                outputs = [
                    model(img[0].cuda(non_blocking=True))
                    for img in pack['img']
                ]
            else:
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
                os.path.join(cam_out_dir, img_name + '.npy'), {
                    "keys": valid_cat,
                    "cam": strided_cam.cpu(),
                    "high_res": highres_cam.cpu().numpy()
                })

            if iter % (len(dataset) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(dataset) // 20)), end='')


def run(args, cam_weights_name=None, cam_out_dir=None):
    model = CAM()
    model.load_state_dict(torch.load(cam_weights_name), strict=True)
    model.eval()

    dataset = dataloader.VOC12ClassificationDatasetMSF(
        os.path.join(args.dataset_dir, 'val.txt'),
        image_dir=args.image_dir,
        scales=args.cam_scales)

    print('[ ', end='')
    _work(model, dataset, args, cam_out_dir)
    print(']')

    if use_cuda:
        torch.cuda.empty_cache()
