import argparse
import os
import shutil

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from wsl_survey.acol import my_optim
from wsl_survey.datasets.classification_dataset import data_loader
from wsl_survey.acol.models import initialize_model
from wsl_survey.acol.utils import AverageMeter
from wsl_survey.acol.utils import metrics
from wsl_survey.acol.utils.restore import restore
from wsl_survey.acol.utils.save_atten import SAVE_ATTEN

LR = 0.001
# LR=0.1
EPOCH = 200
DISP_INTERVAL = 50


def get_arguments():
    parser = argparse.ArgumentParser(description='ACoL')
    parser.add_argument("--image_dir",
                        type=str,
                        default='',
                        help='Directory of training images')
    parser.add_argument('--image_size',
                        '-i',
                        default=224,
                        type=int,
                        metavar='N',
                        help='image size (default: 224)')
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--arch", type=str, default='vgg_v1')
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epochs", type=int, default=EPOCH)
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=bool, default=False)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--checkpoints", type=str)

    return parser.parse_args()


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.checkpoints, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath,
                        os.path.join(args.checkpoints, 'model_best.pth.tar'))


use_gpu = torch.cuda.is_available()


def get_model(args, num_classes):
    model = initialize_model(args.arch).model(num_classes=num_classes,
                                              args=args,
                                              threshold=args.threshold)
    if use_gpu:
        model.cuda()
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, range(args.num_gpu))

    optimizer = my_optim.get_finetune_optimizer(args, model)

    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=False)
    return model, optimizer


def val(args, model=None, current_epoch=0):
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1.reset()
    top5.reset()
    val_loader = data_loader(args, split_type='val')
    save_bins_path = os.path.join(args.checkpoints, 'save_bins')
    os.makedirs(save_bins_path, exist_ok=True)
    if model is None:
        model, _ = get_model(args, val_loader.dataset.get_number_classes())
    model.eval()

    save_atten = SAVE_ATTEN(save_dir=save_bins_path)

    global_counter = 0
    prob = None
    gt = None
    for idx, dat in tqdm(enumerate(val_loader)):
        img_path, img, label_in = dat
        global_counter += 1
        if args.tencrop == 'True':
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)
            label_input = label_in.repeat(10, 1)
            label = label_input.view(-1)
        else:
            label = label_in
        if use_gpu:
            img, label = img.cuda(), label.cuda()
        img_var, label_var = Variable(img), Variable(label)

        logits = model(img_var, label_var)

        logits0 = logits[0]
        if args.tencrop == 'True':
            logits0 = logits0.view(bs, ncrops, -1).mean(1)

        prec1_1, prec5_1 = metrics.accuracy(logits0.cpu().data,
                                            label_in.long(),
                                            topk=(1, 5))
        # prec3_1, prec5_1 = Metrics.accuracy(logits[1].data, label.long(), topk=(1,5))
        top1.update(prec1_1[0], img.size()[0])
        top5.update(prec5_1[0], img.size()[0])

        # model.module.save_erased_img(img_path)
        last_featmaps = model.module.get_localization_maps()
        np_last_featmaps = last_featmaps.cpu().data.numpy()

        # Save 100 sample masked images by heatmaps
        if idx < 100 / args.batch_size:
            save_atten.get_masked_img(img_path,
                                      np_last_featmaps,
                                      label_in.numpy(),
                                      size=(0, 0),
                                      maps_in_dir=False)

        # save_atten.get_masked_img(img_path, np_last_featmaps, label_in.numpy(),size=(0,0),
        #                           maps_in_dir=True, save_dir='../heatmaps',only_map=True )

        # np_scores, pred_labels = torch.topk(logits0,k=args.num_classes,dim=1)
        # pred_np_labels = pred_labels.cpu().data.numpy()
        # save_atten.save_top_5_pred_labels(pred_np_labels[:,:5], img_path, global_counter)
        # # pred_np_labels[:,0] = label.cpu().numpy() #replace the first label with gt label
        # # save_atten.save_top_5_atten_maps(np_last_featmaps, pred_np_labels, img_path)

    if args.onehot:
        print(val_mAP)
        print('AVG:', np.mean(val_mAP))

    else:
        print('Top1:', top1.avg, 'Top5:', top5.avg)

    # save_name = os.path.join(args.checkpoints, 'val_result.txt')
    # with open(save_name, 'a') as f:
    #     f.write('%.3f'%out)


if __name__ == '__main__':
    args = get_arguments()
    import json

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    os.makedirs(args.checkpoints, exist_ok=True)
    val(args)
