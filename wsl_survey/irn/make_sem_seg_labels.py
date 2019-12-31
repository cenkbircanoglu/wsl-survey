import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

from wsl_survey.irn import dataloader, indexing
from wsl_survey.irn.net.resnet50_irn import EdgeDisplacement

use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.enabled = True


def _work(model, dataset, args, sem_seg_out_dir, cam_out_dir):
    data_loader = DataLoader(dataset,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=False)

    with torch.no_grad():
        if use_cuda:
            model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            orig_img_size = np.asarray(pack['size'])

            if use_cuda:
                edge, dp = model(pack['img'][0].cuda(non_blocking=True))
            else:
                edge, dp = model(pack['img'][0])
            cam_dict = np.load(cam_out_dir + '/' + img_name + '.npy',
                               allow_pickle=True).item()

            cam_downsized_values = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            if use_cuda:
                cam_downsized_values = cam_downsized_values.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values,
                                            edge,
                                            beta=args.beta,
                                            exp_times=args.exp_times,
                                            radius=5)

            rw_up = F.interpolate(
                rw, scale_factor=4, mode='bilinear',
                align_corners=False)[...,
                    0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0),
                             value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            imageio.imsave(os.path.join(sem_seg_out_dir, img_name + '.png'),
                           rw_pred.astype(np.uint8))

            if iter % (len(dataset) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(dataset) // 20)), end='')


def run(args, irn_weights_name=None, sem_seg_out_dir=None, cam_out_dir=None):
    model = EdgeDisplacement()
    model.load_state_dict(torch.load(irn_weights_name), strict=False)
    model.eval()

    dataset = dataloader.VOC12ClassificationDatasetMSF(
        os.path.join(args.dataset_dir, 'train.txt'),
        image_dir=args.image_dir,
        scales=(1.0,))

    print("[", end='')
    _work(model, dataset, args, sem_seg_out_dir, cam_out_dir)
    print("]")
    if use_cuda:
        torch.cuda.empty_cache()
