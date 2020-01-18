import importlib
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing, cuda
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsl_survey.segmentation.irn.misc import torchutils, indexing
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
            img_name = dataloader.decode_int_filename(pack['name'][0])
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0])

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy',
                               allow_pickle=True).item()

            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams

            rw = indexing.propagate_to_edge(cam_downsized_values, edge,
                                            beta=args.beta,
                                            exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear',
                                  align_corners=False)[..., 0,
                    :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0),
                             value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            imageio.imsave(
                os.path.join(args.sem_seg_out_dir, img_name + '.png'),
                rw_pred.astype(np.uint8))

            if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def _work_gpu(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False,
                             num_workers=args.num_workers // n_gpus,
                             pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in tqdm(enumerate(data_loader), total=len(databin)):
            img_name = dataloader.decode_int_filename(pack['name'][0])
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy',
                               allow_pickle=True).item()

            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge,
                                            beta=args.beta,
                                            exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear',
                                  align_corners=False)[..., 0,
                    :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0),
                             value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            imageio.imsave(
                os.path.join(args.sem_seg_out_dir, img_name + '.png'),
                rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network),
                    'EdgeDisplacement')()
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    dataset = dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
                                                       voc12_root=args.voc12_root,
                                                       scales=(1.0,),
                                                       class_label_dict_path=args.class_label_dict_path)
    print("[", end='')
    if use_gpu:
        n_gpus = torch.cuda.device_count()

        dataset = torchutils.split_dataset(dataset, n_gpus)

        multiprocessing.spawn(_work_gpu, nprocs=n_gpus,
                              args=(model, dataset, args),
                              join=True)
    else:
        dataset = torchutils.split_dataset(dataset, args.num_workers)
        multiprocessing.spawn(_work_cpu, nprocs=args.num_workers,
                              args=(model, dataset, args),
                              join=True)
    print("]")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./data/test/VOC2012',
        class_label_dict_path='./data/test/VOC2012/ImageSets/Segmentation/cls_labels.npy',
        train_list='./data/test/VOC2012/ImageSets/Segmentation/train_aug.txt',
        ir_label_out_dir='./outputs/test/results/resnet18/irn_label',
        infer_list='./data/voc12/train.txt',
        irn_network='ResNet18',
        irn_num_epoches=1,
        irn_batch_size=4
    )
    args = parser.parse_args()
    run(args)
