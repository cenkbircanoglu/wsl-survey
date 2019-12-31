import os

import imageio
import numpy as np
from torch.utils.data import DataLoader

from wsl_survey.irn import imutils, dataloader


def _work(infer_dataset, args, ir_label_out_dir, cam_out_dir):
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        img_name = pack['name'][0]
        img = pack['img'][0].numpy()
        cam_dict = np.load(os.path.join(cam_out_dir, img_name + '.npy'),
                           allow_pickle=True).item()

        cams = cam_dict['high_res']
        keys = np.pad(cam_dict['keys'].cpu() + 1, (1, 0), mode='constant')

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

        imageio.imwrite(os.path.join(ir_label_out_dir, img_name + '.png'),
                        conf.astype(np.uint8))

        if iter % (len(infer_dataset) // 20) == 0:
            print("%d " % ((5 * iter + 1) // (len(infer_dataset) // 20)),
                  end='')


def run(args, ir_label_out_dir=None, cam_out_dir=None):
    dataset = dataloader.VOC12ImageDataset(os.path.join(
        args.dataset_dir, 'train.txt'),
                                           image_dir=args.image_dir,
                                           img_normal=None,
                                           to_torch=False)

    print('[ ', end='')
    _work(dataset, args, ir_label_out_dir, cam_out_dir)
    print(']')
