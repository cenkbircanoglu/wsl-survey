import importlib

import torch
from torch.backends import cudnn
from tqdm import tqdm

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from wsl_survey.segmentation.irn.voc12 import dataloader
from wsl_survey.segmentation.irn.misc import pyutils, torchutils

use_gpu = torch.cuda.is_available()


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label']
            if np.any(label):
                if use_gpu:
                    label = label.cuda(non_blocking=True)
                x = model(img)
                loss1 = F.multilabel_soft_margin_loss(x, label)
                val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


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
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True)
    max_step = (len(train_dataset) * 10 //
                args.cam_batch_size) * args.cam_num_epoches

    val_dataset = dataloader.VOC12ClassificationDataset(
        args.val_list,
        voc12_root=args.voc12_root,
        crop_size=512,
        class_label_dict_path=args.class_label_dict_path)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=args.cam_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    model = getattr(importlib.import_module(args.cam_network_module),
                    args.cam_network)(
        num_classes=train_dataset.label_list[0].shape[0])

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {
            'params': param_groups[0],
            'lr': args.cam_learning_rate,
            'weight_decay': args.cam_weight_decay
        },
        {
            'params': param_groups[1],
            'lr': 10 * args.cam_learning_rate,
            'weight_decay': args.cam_weight_decay
        },
    ],
        lr=args.cam_learning_rate,
        weight_decay=args.cam_weight_decay,
        max_step=max_step)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.cam_num_epoches))

        for step, pack in tqdm(enumerate(train_data_loader),
                               total=len(train_dataset) //
                                     args.cam_batch_size):

            img = pack['img']
            label = pack['label']

            if use_gpu:
                label = label.cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size /
                                     timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()),
                      flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()
    try:
        state_dict = model.module.state_dict()
    except:
        state_dict = model.state_dict()
    torch.save(state_dict, args.cam_weights_name + '.pth')
    if use_gpu:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    from wsl_survey.segmentation.irn.config import make_parser
    import os

    parser = make_parser()
    parser.set_defaults(
        voc12_root='./data/test1/VOC2012',
        class_label_dict_path='./data/voc12/cls_labels.npy',
        train_list='./data/test1/VOC2012/ImageSets/Segmentation/train_aug.txt',
        val_list='./data/test1/VOC2012/ImageSets/Segmentation/val.txt',
        cam_weights_name='./outputs/test1/results/resnet18/sess/cam.pth',
        cam_network='ResNet18',
        cam_network_module='wsl_survey.segmentation.irn.net.resnet_cam',
        cam_num_epoches=1,
        cam_batch_size=4)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.cam_weights_name), exist_ok=True)
    run(args)
