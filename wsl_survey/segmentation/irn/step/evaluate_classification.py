import importlib

import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
from wsl_survey.segmentation.irn.voc12 import dataloader
from wsl_survey.segmentation.irn.misc import pyutils

use_gpu = torch.cuda.is_available()


def run(args):
    assert args.voc12_root is not None
    assert args.class_label_dict_path is not None
    assert args.train_list is not None
    assert args.val_list is not None
    assert args.cam_weights_name is not None
    assert args.cam_network is not None
    assert args.cam_num_epoches is not None
    assert args.cam_network_module is not None

    train_dataset = dataloader.VOC12ClassificationDataset(
        args.train_list,
        voc12_root=args.voc12_root,
        resize_long=(320, 640),
        hor_flip=True,
        crop_size=512,
        crop_method="random",
        class_label_dict_path=args.class_label_dict_path)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.cam_batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   drop_last=False)
    model = getattr(importlib.import_module(args.cam_network_module),
                    args.cam_network)(
        num_classes=train_dataset.label_list[0].shape[0])

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    model.eval()

    avg_meter = pyutils.AverageMeter()
    total = 0
    correct = 0

    with torch.no_grad():
        for pack in train_data_loader:

            img = pack['img']
            label = pack['label']

            if use_gpu:
                label = label.cuda(non_blocking=True)
            x = model(img)

            predicted = torch.argmax(x, dim=1)
            label = torch.argmax(label, dim=1)
            total += len(label)
            correct += (predicted.numpy() == label.numpy().astype(int)).sum()
            train_accuracy = 100 * correct / total

    print('acc:%.4f' % (train_accuracy), flush=True)


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser
    import os

    subset = 'subset10'

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./datasets/voc2012/VOCdevkit/VOC2012',
        class_label_dict_path='./data/voc12/%s/cls_labels.npy' % subset,
        train_list='./data/voc12/%s/train.txt' % subset,
        cam_weights_name='./results/%s_resnet152/%s_resnet152/sess/cam.pth' % (
            subset, subset),
        cam_network='ResNet152',
        cam_network_module='wsl_survey.segmentation.irn.net.resnet_cam',
        cam_batch_size=16)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.cam_weights_name), exist_ok=True)
    run(args)
