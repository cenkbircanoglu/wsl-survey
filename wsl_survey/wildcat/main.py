import os

import torch
import torch.nn as nn

from wsl_survey.datasets.classification_dataset import data_loader
from wsl_survey.wildcat.engine import MultiLabelMAPEngine
from wsl_survey.wildcat.models import resnet101_wildcat


def train(args):
    use_gpu = torch.cuda.is_available()

    # define dataset
    train_loader = data_loader(args, split_type='train')
    val_loader = data_loader(args, split_type='val')

    num_classes = train_loader.dataset.get_number_classes()

    # load model
    model = resnet101_wildcat(num_classes,
                              pretrained=True,
                              kmax=args.k,
                              alpha=args.alpha,
                              num_maps=args.maps)
    print('classifier', model.classifier)
    print('spatial pooling', model.spatial_pooling)

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'max_epochs': args.epochs,
        'evaluate': args.evaluate,
        'resume': args.resume,
        'use_gpu': use_gpu,
        'difficult_examples': False,
        'save_model_path': args.checkpoints
    }

    engine = MultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_loader, val_loader, optimizer)


if __name__ == '__main__':
    import warnings
    import argparse

    warnings.simplefilter("ignore", UserWarning)

    parser = argparse.ArgumentParser(description='WILDCAT Training')

    parser.add_argument('--dataset_dir',
                        metavar='DIR',
                        help='path to dataset (e.g. ../data/')
    parser.add_argument('--image_dir',
                        metavar='DIR',
                        help='path to dataset (e.g. ../data/')
    parser.add_argument("--checkpoints", type=str)
    parser.add_argument('--image_size',
                        '-i',
                        default=224,
                        type=int,
                        metavar='N',
                        help='image size (default: 224)')
    parser.add_argument('-j',
                        '--num_workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--lrp',
                        '--learning-rate-pretrained',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='learning rate for pre-trained layers')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay',
                        '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq',
                        '-p',
                        default=0,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--k',
                        default=1,
                        type=float,
                        metavar='N',
                        help='number of regions (default: 1)')
    parser.add_argument('--alpha',
                        default=1,
                        type=float,
                        metavar='N',
                        help='weight for the min regions (default: 1)')
    parser.add_argument('--maps',
                        default=1,
                        type=int,
                        metavar='N',
                        help='number of maps per class (default: 1)')
    parser.add_argument("--onehot", type=bool, default=True)
    args = parser.parse_args()

    os.makedirs(args.checkpoints, exist_ok=True)
    train(args)
