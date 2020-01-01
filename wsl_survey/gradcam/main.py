from __future__ import print_function, division

import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models

from wsl_survey.datasets.classification_dataset import data_loader
from wsl_survey.gradcam.networks import *

parser = argparse.ArgumentParser(
    description='PyTorch Digital Mammography Training')
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
parser.add_argument("--checkpoints", type=str)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lr_decay_epoch',
                    default=20,
                    type=int,
                    help='learning decay epoch')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--weight_decay',
                    default=5e-4,
                    type=float,
                    help='weight decay')
parser.add_argument('--finetune',
                    '-f',
                    action='store_true',
                    help='Fine tune pretrained model')
parser.add_argument('--add_layer',
                    '-a',
                    action='store_true',
                    help='Add additional layer in fine-tuning')
parser.add_argument('--reset_classifier',
                    '-r',
                    action='store_true',
                    help='Reset classifier')
parser.add_argument('--test_only',
                    '-t',
                    action='store_true',
                    help='Test mode with the saved model')
parser.add_argument("--onehot", type=bool, default=False)
parser.add_argument("--num_workers", type=int, default=20)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--feature_size", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=64)

args = parser.parse_args()

os.makedirs(args.checkpoints, exist_ok=True)
# Phase 1 : Data Upload
print('\n[Phase 1] : Data Preperation')

dataset_loaders = {
    x: data_loader(args, split_type=x)
    for x in ['train', 'val']
}

use_gpu = torch.cuda.is_available()

# Phase 2 : Model setup
print('\n[Phase 2] : Model setup')


def get_network(args):
    if args.net_type == 'alexnet':
        net = models.alexnet(pretrained=args.finetune)
        file_name = 'alexnet'
    elif args.net_type == 'vggnet':
        if args.depth == 11:
            net = models.vgg11(pretrained=args.finetune)
        elif args.depth == 13:
            net = models.vgg13(pretrained=args.finetune)
        elif args.depth == 16:
            net = models.vgg16(pretrained=args.finetune)
        elif args.depth == 19:
            net = models.vgg19(pretrained=args.finetune)
        else:
            print(
                'Error : VGGnet should have depth of either [11, 13, 16, 19]')
            sys.exit(1)
        file_name = 'vgg-%s' % (args.depth)
    elif args.net_type == 'resnet':
        net = resnet(args.finetune, args.depth)

        file_name = 'resnet-%s' % (args.depth)
    else:
        print(
            'Error : Network should be either [alexnet / vggnet / resnet / densenet]'
        )
        sys.exit(1)

    return net, file_name


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# Test only option
if args.test_only:
    print("| Loading checkpoint model for test phase...")
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = get_network(args)
    print('| Loading ' + file_name + ".t7...")
    checkpoint = torch.load(os.path.join(args.checkpoints, file_name + '.t7'))
    model = checkpoint['model']

    if use_gpu:
        model.cuda()
        model = torch.nn.DataParallel(model,
                                      device_ids=range(
                                          torch.cuda.device_count()))
        cudnn.benchmark = True

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_loader = data_loader(args, split_type='test')

    print("\n[Phase 3 : Inference")
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)

        print(outputs.data.cpu().numpy()[0])
        file_name = 'densenet-%s' % (args.depth)
        softmax_res = softmax(outputs.data.cpu().numpy()[0])

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100. * correct / total
    print("| Test Result\tAcc@1 %.2f%%" % (acc))

    sys.exit(0)


# Training model
def train_model(model,
                criterion,
                optimizer,
                lr_scheduler,
                num_epochs=args.epochs):
    global dataset_dir
    since = time.time()

    best_model, best_acc = model, 0.0

    print('\n[Phase 3] : Training Model')
    print('| Training Epochs = %d' % num_epochs)
    print('| Initial Learning Rate = %f' % args.lr)
    print('| Optimizer = SGD')
    output_file = os.path.join(args.checkpoints, "logs",
                               args.net_type + ".csv")
    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, 'w') as csvfile:
        fields = ['epoch', 'train_acc', 'val_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        for epoch in range(num_epochs):
            train_acc = 0
            val_acc = 0
            for phase in ['train', 'val']:
                loader = dataset_loaders[phase]
                if phase == 'train':
                    optimizer, lr = lr_scheduler(optimizer, epoch)
                    print('\n=> Training Epoch #%d, LR=%f' % (epoch + 1, lr))
                    model.train(True)
                else:
                    model.train(False)
                    model.eval()

                running_loss, running_corrects, tot = 0.0, 0, 0

                for batch_idx, (inputs, labels) in enumerate(loader):
                    labels = labels.to(torch.long)
                    if use_gpu:
                        inputs, labels = Variable(inputs[0].cuda()), Variable(
                            labels.cuda())
                    else:
                        inputs, labels = Variable(inputs[0]), Variable(labels)

                    optimizer.zero_grad()

                    # Forward Propagation
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # Backward Propagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Statistics
                    running_loss += loss.data.item()
                    running_corrects += preds.eq(labels.data).cpu().sum()
                    tot += labels.size(0)

                    if (phase == 'train'):
                        sys.stdout.write('\r')
                        sys.stdout.write(
                            '| Epoch [%2d/%2d] Iter [%3d/%3d]\t\tLoss %.4f\tAcc %.2f%%'
                            %
                            (epoch + 1, num_epochs, batch_idx + 1,
                             (len(loader.dataset) // args.batch_size) + 1,
                             loss.data.item(), 100. * running_corrects / tot))
                        sys.stdout.flush()
                        sys.stdout.write('\r')

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc = float(running_corrects) / len(loader.dataset)

                if phase == 'train':
                    train_acc = epoch_acc

                if phase == 'val':
                    print(
                        '\n| Validation Epoch #%d\t\t\tLoss %.4f\tAcc %.2f%%' %
                        (epoch + 1, loss.data.item(), 100. * epoch_acc))

                    if epoch_acc > best_acc:
                        print('| Saving Best model...\t\t\tTop1 %.2f%%' %
                              (100. * epoch_acc))
                        best_acc = epoch_acc
                        best_model = copy.deepcopy(model)
                        state = {
                            'model': best_model,
                            'acc': epoch_acc,
                            'epoch': epoch,
                        }
                        torch.save(state, os.path.join(args.checkpoints,
                                                       file_name + '.t7'))

                    val_acc = epoch_acc

            writer.writerow({
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'val_acc': val_acc
            })

    csvfile.close()
    time_elapsed = time.time() - since
    print('\nTraining completed in\t{:.0f} min {:.0f} sec'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc\t{:.2f}%'.format(best_acc * 100))

    return best_model


def exp_lr_scheduler(optimizer,
                     epoch,
                     init_lr=args.lr,
                     weight_decay=args.weight_decay,
                     lr_decay_epoch=args.lr_decay_epoch):
    lr = init_lr * (0.5 ** (epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr


model_ft, file_name = get_network(args)

if args.reset_classifier:
    print('| Reset final classifier...')
    if args.add_layer:
        print('| Add features of size %d' % args.feature_size)
        num_ftrs = model_ft.fc.in_features
        feature_model = list(model_ft.fc.children())
        feature_model.append(nn.Linear(num_ftrs, args.feature_size))
        feature_model.append(nn.BatchNorm1d(args.feature_size))
        feature_model.append(nn.ReLU(inplace=True))
        feature_model.append(
            nn.Linear(args.feature_size,
                      len(dataset_loaders['train'].dataset.classes)))
        model_ft.fc = nn.Sequential(*feature_model)
    else:
        if args.net_type == 'alexnet' or args.net_type == 'vggnet':
            num_ftrs = model_ft.classifier[6].in_features
            feature_model = list(model_ft.classifier.children())
            feature_model.pop()
            feature_model.append(
                nn.Linear(num_ftrs,
                          len(dataset_loaders['train'].dataset.classes)))
            model_ft.classifier = nn.Sequential(*feature_model)
        elif args.net_type == 'resnet':
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(
                num_ftrs, len(dataset_loaders['train'].dataset.classes))

if use_gpu:
    model_ft = model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft,
                                     device_ids=range(
                                         torch.cuda.device_count()))
    cudnn.benchmark = True

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(),
                             lr=args.lr,
                             momentum=0.9,
                             weight_decay=args.weight_decay)
    model_ft = train_model(model_ft,
                           criterion,
                           optimizer_ft,
                           exp_lr_scheduler,
                           num_epochs=args.epochs)
