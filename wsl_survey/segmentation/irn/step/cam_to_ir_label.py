import os

import cv2
import imageio
import numpy as np
from torch import multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsl_survey.segmentation.irn.misc import torchutils, imutils
from wsl_survey.segmentation.irn.voc12 import dataloader


def generate_bbox(path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_img = cv2.imread(path)
    ret, threshed_img = cv2.threshold(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(threshed_img, cv2.MORPH_CLOSE, kernel, iterations=8)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_area = float('-inf')
    x, y, w, h = None, None, None, None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > biggest_area:
            biggest_area = area
            x, y, w, h = cv2.boundingRect(cnt)
    with open(output_path, mode='w') as f:
        f.write('%s\t%s\t%s\t%s\n' % (x, y, w, h))


def _work(process_id, infer_dataset, args):
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)

    for iter, pack in tqdm(enumerate(infer_data_loader), total=len(databin)):
        try:
            img_name = dataloader.decode_int_filename(pack['name'][0])
            path = os.path.join(args.ir_label_out_dir, img_name + '.png')
            bbox_path = os.path.join(args.bbox_out_dir, img_name + '.txt')
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                img = pack['img'][0].numpy()
                cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'),
                                   allow_pickle=True).item()

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

                imageio.imwrite(path, conf.astype(np.uint8))

                if process_id == args.num_workers - 1 and iter % (len(databin) //
                                                                  20) == 0:
                    print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')
            generate_bbox(path, bbox_path)
        except Exception as e:
            print(e)

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
