import argparse
import json
import os
import shutil
import time

import torch
from tqdm import tqdm

from wsl_survey.acol import my_optim
from wsl_survey.acol.models import initialize_model
from wsl_survey.acol.utils import AverageMeter
from wsl_survey.acol.utils import metrics
from wsl_survey.acol.utils.restore import restore
# Default parameters
from wsl_survey.datasets.classification_dataset import data_loader

LR = 0.0001
DISP_INTERVAL = 20


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
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--arch", type=str, default='vgg_v0')
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_gpu", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--checkpoints", type=str)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=bool, default=False)
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()


use_gpu = torch.cuda.is_available()


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.checkpoints, filename)
    os.makedirs(args.checkpoints, exist_ok=True)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath,
                        os.path.join(args.checkpoints, 'model_best.pth.tar'))


def get_model(args, num_classes):
    model = initialize_model(args.arch).model(pretrained=False,
                                              num_classes=num_classes,
                                              threshold=args.threshold,
                                              args=args)
    if use_gpu:
        model.cuda()
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, range(args.num_gpu))

    optimizer = my_optim.get_finetune_optimizer(args, model)

    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=False)
    return model, optimizer


def train(args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loader = data_loader(args, split_type='train')
    model, optimizer = get_model(args,
                                 train_loader.dataset.get_number_classes())
    model.train()

    with open(os.path.join(args.checkpoints, 'train_record.csv'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.write('#epoch,loss,pred@1,pred@5\n')

    total_epoch = args.epochs
    global_counter = args.global_counter
    current_epoch = args.current_epoch
    end = time.time()
    max_iter = total_epoch * len(train_loader)
    print('Max iter:', max_iter)
    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        top1.reset()
        top5.reset()
        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)

        if res:
            for g in optimizer.param_groups:
                out_str = 'Epoch:%d, %f\n' % (current_epoch, g['lr'])
                fw.write(out_str)

        for idx, dat in enumerate(train_loader):
            img, label = dat
            global_counter += 1
            label = torch.tensor(list(label))
            img = img[0]
            if use_gpu:
                img, label = img.cuda(), label.cuda()
            img_var = torch.autograd.Variable(img)
            label_var = torch.autograd.Variable(label)
            logits = model(img_var, label_var)
            loss_val, = model.get_loss(logits, label_var)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if not args.onehot:
                logits1 = torch.squeeze(logits[0])
                prec1_1, prec5_1 = metrics.accuracy(logits1.data,
                                                    label.long(),
                                                    topk=(1, 5))
                top1.update(prec1_1[0], img.size()[0])
                top5.update(prec5_1[0], img.size()[0])

            losses.update(loss_val.data.item(), img.size()[0])
            batch_time.update(time.time() - end)

            end = time.time()
            if global_counter % 1000 == 0:
                losses.reset()
                top1.reset()
                top5.reset()

            if global_counter % args.disp_interval == 0:
                # Calculate ETA
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          current_epoch,
                          global_counter % len(train_loader),
                          len(train_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))

        if current_epoch % 1 == 0:
            save_checkpoint(args, {
                'epoch': current_epoch,
                'arch': 'resnet',
                'global_counter': global_counter,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                            is_best=False,
                            filename='%s_epoch_%d_glo_step_%d.pth.tar' %
                            (args.dataset_dir.replace(
                                '/', '_'), current_epoch, global_counter))

        with open(os.path.join(args.checkpoints, 'train_record.csv'),
                  'a') as fw:
            fw.write('%d,%.4f,%.3f,%.3f\n' %
                     (current_epoch, losses.avg, top1.avg, top5.avg))

        current_epoch += 1


if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    os.makedirs(args.checkpoints, exist_ok=True)
    train(args)
