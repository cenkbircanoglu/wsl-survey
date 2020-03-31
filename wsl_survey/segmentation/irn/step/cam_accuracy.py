import importlib
import os

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsl_survey.segmentation.irn.voc12 import dataloader

cudnn.enabled = True
use_gpu = torch.cuda.is_available()


def _work_cpu_1(model, dataset, args):
    data_loader = DataLoader(dataset,
                             batch_size=args.cam_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False)

    with torch.no_grad():
        correct = 0.
        total = 0.
        for iter, pack in tqdm(enumerate(data_loader), total=len(dataset) // args.cam_batch_size):
            try:
                img = pack['img'].cuda()
                label = pack['label']
                if use_gpu:
                    label = label.cuda(non_blocking=True)

                x = model(img)

                _, predicted = torch.max(x.data, 1)
                _, actual = torch.max(label.data, 1)
                correct += (predicted == actual).sum()
                total += label.shape[0]
            except Exception as e:
                print(e)

        acc = 100 * correct / total
        print(acc)


def run(args):
    assert args.voc12_root is not None
    assert args.class_label_dict_path is not None
    assert args.train_list is not None
    assert args.cam_weights_name is not None
    assert args.cam_network is not None
    assert args.cam_out_dir is not None
    assert args.cam_network_module is not None

    model = getattr(importlib.import_module(args.cam_network_module),
                    args.cam_network)(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    if use_gpu:
        model.cuda()
    model.eval()
    dataset = dataloader.VOC12ClassificationDataset(
        args.infer_list,
        voc12_root=args.voc12_root,
        resize_long=(320, 640),
        hor_flip=True,
        crop_size=512,
        crop_method="random",
        class_label_dict_path=args.class_label_dict_path)
    print('[ ', end='')
    _work_cpu_1(model, dataset, args)
    print(']')


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
