import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class Relu(nn.Module):
    def forward(self, x):
        return F.relu(x, inplace=True)


class AvgPooling(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))


class Mean(nn.Module):
    def forward(self, x):
        return x.mean([2, 3])


def network_features(net, version):
    if version == 'alexnet':
        features = nn.Sequential(getattr(net, 'features'), getattr(net, 'avgpool'), Flatten())
        feature_size = net.classifier[1].in_features

    if version.startswith('densenet'):
        features = nn.Sequential(getattr(net, 'features'), Relu(), AvgPooling(), Flatten())
        feature_size = net.classifier.in_features

    if version.startswith('mnasnet'):
        features = nn.Sequential(getattr(net, 'layers'), Mean())
        feature_size = net.classifier[1].in_features

    if version.startswith('mobilenet'):
        features = nn.Sequential(getattr(net, 'features'), Mean())
        feature_size = net.classifier[1].in_features

    if version.startswith('resne'):
        features = nn.Sequential(getattr(net, 'conv1'), getattr(net, 'bn1'), getattr(net, 'relu'),
                                 getattr(net, 'maxpool'), getattr(net, 'layer1'), getattr(net, 'layer2'),
                                 getattr(net, 'layer3'), getattr(net, 'layer4'), getattr(net, 'avgpool'), Flatten())
        feature_size = net.fc.in_features

    if version.startswith('shufflenet'):
        features = nn.Sequential(getattr(net, 'conv1'),
                                 getattr(net, 'maxpool'), getattr(net, 'stage2'), getattr(net, 'stage3'),
                                 getattr(net, 'stage4'), getattr(net, 'conv5'), Mean())
        feature_size = net.fc.in_features

    if version.startswith('vgg'):
        features = nn.Sequential(getattr(net, 'features'), getattr(net, 'avgpool'), Flatten())
        feature_size = net.classifier[0].in_features

    if version.startswith('wide_resnet'):
        features = nn.Sequential(getattr(net, 'conv1'), getattr(net, 'bn1'), getattr(net, 'relu'),
                                 getattr(net, 'maxpool'), getattr(net, 'layer1'), getattr(net, 'layer2'),
                                 getattr(net, 'layer3'), getattr(net, 'layer4'), getattr(net, 'avgpool'), Flatten())
        feature_size = net.fc.in_features

    return features, feature_size
