from __future__ import division
from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from wsl_survey.base.nets.multi_output_networks import load_multi_output_network
from wsl_survey.base.train_multi_output_network import train_multi_output_network
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
parser.add_argument("--model_file", type=str)

args = parser.parse_args()

dataloaders_dict = {
    'train': data_loader(args, 'train'),
    'val': data_loader(args, 'test')
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

category_list = ['make_id', 'model_id', 'year']

feature_name_size = {
    category_name: dataloaders_dict['train'].dataset.get_number_classes(category_name) for category_name in
    category_list
}

model_ft = load_multi_output_network(args.network_name, pretrained=True, feature_name_size=feature_name_size,
                                     finetune=args.finetune, image_size=args.image_size)

print(model_ft)

model_ft = model_ft.to(device)

params_to_update = model_ft.trainable_parameters()
optimizer_ft = optim.AdamW(params_to_update)

criterion = nn.CrossEntropyLoss()


def parse_input_fn(inputs, device):
    return inputs[0].to(device)


def parse_label_fn(labels, device):
    return [labels[category_name].long().to(device) for category_name in category_list]


model_ft, hist = train_multi_output_network(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=args.epochs,
                                            is_inception=(args.network_name == "inception"),
                                            parse_input_fn=parse_input_fn, parse_label_fn=parse_label_fn,
                                            category_list=category_list)

try:
    state_dict = model_ft.module.state_dict()
except:
    state_dict = model_ft.state_dict()

os.makedirs(os.path.dirname(args.model_file), exist_ok=True)
torch.save(state_dict, args.model_file + '.pth')
ohist = [h.cpu().numpy() for h in hist]
print(args.model_file, ohist)
