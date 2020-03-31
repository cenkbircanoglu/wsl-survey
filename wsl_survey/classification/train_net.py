import importlib

import torch
from torch.backends import cudnn
from tqdm import tqdm

from wsl_survey.datasets.compcars.classification_dataset import data_loader

cudnn.enabled = True
import torch.nn.functional as F

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
            if use_gpu:
                label = label.cuda(non_blocking=True)
            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return


def run(args):
    assert args.train_list is not None
    assert args.val_list is not None
    assert args.weights_name is not None
    assert args.network is not None
    assert args.num_epoches is not None
    assert args.network_module is not None

    train_data_loader = data_loader(args, 'train')
    max_step = (len(train_data_loader.dataset) // args.batch_size) * args.num_epoches
    num_classes_list = train_data_loader.dataset.get_number_classes()
    model = getattr(importlib.import_module(args.network_module), args.network)(pretrained=True,
                                                                                num_classes_list=num_classes_list)

    val_data_loader = data_loader(args, 'test')

    params = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {
            'params': params[0],
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay
        },
        {
            'params': params[1],
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay
        },
        {
            'params': params[2],
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay
        }
    ], lr=args.learning_rate, weight_decay=args.weight_decay, max_step=max_step)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    for ep in range(args.num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.num_epoches))
        correct = {'make': 0.0, 'model': 0.0, 'year': 0.0}
        total = 0.
        for step, pack in tqdm(enumerate(train_data_loader),
                               total=len(train_data_loader.dataset) // args.batch_size):

            (img, path), (make_label, model_label, year_label) = pack
            if use_gpu:
                make_label = make_label.cuda(non_blocking=True)
                model_label = model_label.cuda(non_blocking=True)
                year_label = year_label.cuda(non_blocking=True)

            x = model(img)
            make_logit = x[0].data
            model_logit = x[1].data
            year_logit = x[2].data

            _, make_predicted = torch.max(make_logit, 1)
            _, mode_predicted = torch.max(model_logit, 1)
            _, year_predicted = torch.max(year_logit, 1)

            _, make_actual = torch.max(make_label.data, 1)
            _, model_actual = torch.max(model_label.data, 1)
            _, year_actual = torch.max(year_label.data, 1)

            correct['make'] += (make_predicted == make_actual).sum()
            correct['model'] += (mode_predicted == model_actual).sum()
            correct['year'] += (year_predicted == year_actual).sum()

            total += make_label.shape[0]

            make_loss = F.multilabel_soft_margin_loss(x[0], make_label)
            model_loss = F.multilabel_soft_margin_loss(x[1], model_label)
            year_loss = F.multilabel_soft_margin_loss(x[2], year_label)

            loss = make_loss + model_loss + year_loss

            avg_meter.add({'make_loss': make_loss.item()})
            avg_meter.add({'model_loss': model_loss.item()})
            avg_meter.add({'year_loss': year_loss.item()})

            optimizer.zero_grad()

            avg_meter.add({'loss': loss.item()})
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                make_acc = 100 * correct['make'] / total
                model_acc = 100 * correct['model'] / total
                year_acc = 100 * correct['year'] / total
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((step + 1) * args.batch_size /
                                     timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()),
                      'make_acc:%s' % make_acc,
                      'model_acc:%s' % model_acc,
                      'year_acc:%s' % year_acc,
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default=299, type=int)
    parser.add_argument("--onehot", default=True, type=bool)
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--dataset_dir",
                        default='/Users/cenk.bircanoglu/workspace/icpr/arxiv_data/train_test_split/classification',
                        type=str)
    parser.add_argument("--image_dir", default='/Users/cenk.bircanoglu/workspace/icpr/', type=str)

    # Class Activation Map
    parser.add_argument("--network", type=str)
    parser.add_argument("--network_module", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_epoches", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--eval_thres", default=0.15, type=float)
    parser.add_argument("--scales",
                        default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument(
        "--exp_times",
        default=8,
        help=
        "Hyper-parameter that controls the number of random walk iterations,"
        "The random walk is performed 2^{exp_times}.")
    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--weights_name", type=str)

    parser.set_defaults(
        weights_name='./outputs/test1/results/resnet18/sess/cam.pth',
        network='resnet18',
        network_module='wsl_survey.classification.resnet')
    args = parser.parse_args()
    run(args)
