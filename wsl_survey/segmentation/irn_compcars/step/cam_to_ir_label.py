import os

import imageio
import numpy as np
from torch import multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsl_survey.segmentation.irn.misc import torchutils, imutils
from wsl_survey.segmentation.irn_compcars.compcars import dataloader


def _work(process_id, infer_dataset, args):
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)

    for iter, pack in tqdm(enumerate(infer_data_loader), total=len(databin)):
        img_name = pack['name'][0]
        path = os.path.join(args.ir_label_out_dir, img_name + '.png')
        if not os.path.exists(path):

            img = pack['img'][0].numpy()
            cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

            cams = cam_dict['high_res']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            # 1. find confident fg & bg
            fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)),
                                 mode='constant',
                                 constant_values=args.conf_fg_thres)
            fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
            pred = imutils.crf_inference_label(img,
                                               fg_conf_cam,
                                               n_labels=keys.shape[0])
            fg_conf = keys[pred]

            bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)),
                                 mode='constant',
                                 constant_values=args.conf_bg_thres)
            bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
            pred = imutils.crf_inference_label(img,
                                               bg_conf_cam,
                                               n_labels=keys.shape[0])
            bg_conf = keys[pred]

            # 2. combine confident fg & bg
            conf = fg_conf.copy()
            conf[fg_conf == 0] = 255
            conf[bg_conf + fg_conf == 0] = 0


            os.makedirs(os.path.dirname(path), exist_ok=True)

            imageio.imwrite(path, conf.astype(np.uint8))

            if process_id == args.num_workers - 1 and iter % (len(databin) //
                                                              20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    assert args.voc12_root is not None
    assert args.train_list is not None
    assert args.ir_label_out_dir is not None
    assert args.cam_out_dir is not None

    dataset = dataloader.VOC12ImageDataset(args.train_list,
                                           voc12_root=args.voc12_root,
                                           img_normal=None,
                                           to_torch=False)
    dataset = torchutils.split_dataset(dataset, args.num_workers)

    print('[ ', end='')
    multiprocessing.spawn(_work,
                          nprocs=args.num_workers,
                          args=(dataset, args),
                          join=True)
    print(']')


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./data/test1/VOC2012',
        train_list='./data/test1/VOC2012/ImageSets/Segmentation/train_aug.txt',
        ir_label_out_dir='./outputs/test1/results/resnet18/irn_label',
        cam_out_dir='./outputs/test1/results/resnet18/cam',
        num_workers=1)
    args = parser.parse_args()
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    run(args)
