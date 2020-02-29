from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from wsl_survey.base.nets.networks import load_network
from wsl_survey.base.train_network import train_model
from wsl_survey.compcars.dataset import data_loader

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", default=299, type=int)
parser.add_argument("--finetune", default=False, type=bool)
parser.add_argument("--onehot", default=False, type=bool)
parser.add_argument("--num_workers", default=32, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--dataset_dir", default='/home/ubuntu/icpr/arxiv_data/train_test_split/classification', type=str)
parser.add_argument("--image_dir", default='/home/ubuntu/icpr/', type=str)
parser.add_argument("--network_name", default='resnet18', type=str)
parser.add_argument("--category_name", default='make_id', type=str)
parser.add_argument("--model_file", type=str)

args = parser.parse_args()

dataloaders_dict = {
    'train': data_loader(args, 'train'),
    'val': data_loader(args, 'test')
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = dataloaders_dict['train'].dataset.get_number_classes(args.category_name)

model_ft = load_network(args.network_name, pretrained=True, num_classes=num_classes, finetune=args.finetune,
                        image_size=args.image_size)

print(model_ft)

model_ft = model_ft.to(device)

params_to_update = model_ft.trainable_parameters()
optimizer_ft = optim.Adam(params_to_update)

criterion = nn.CrossEntropyLoss()


def parse_input_fn(inputs):
    return inputs[0]


def parse_label_fn(labels):
    return labels.get(args.category_name)


model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=args.epochs,
                             is_inception=(args.network_name == "inception"), parse_input_fn=parse_input_fn,
                             parse_label_fn=parse_label_fn)

try:
    state_dict = model_ft.module.state_dict()
except:
    state_dict = model_ft.state_dict()
torch.save(state_dict, args.model_file + '.pth')
ohist = [h.cpu().numpy() for h in hist]
print(ohist)
