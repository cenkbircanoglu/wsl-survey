import torch.nn as nn
import torch.nn.functional as F

from wsl_survey.segmentation.irn.misc import torchutils
from wsl_survey.segmentation.irn.net import resnet152


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet101 = resnet152.resnet152(pretrained=True,
                                             strides=[2, 2, 2, 1])

        self.stage1 = nn.Sequential(self.resnet101.conv1,
                                    self.resnet101.bn1,
                                    self.resnet101.relu,
                                    self.resnet101.maxpool,
                                    self.resnet101.layer1)
        self.stage2 = nn.Sequential(self.resnet101.layer2)
        self.stage3 = nn.Sequential(self.resnet101.layer3)
        self.stage4 = nn.Sequential(self.resnet101.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList(
            [self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 20)

        return x

    def train(self, mode=True):
        for p in self.resnet101.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet101.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (
            list(self.backbone.parameters()),
            list(self.newly_added.parameters()))
import torch

class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)
        #print(x.shape)
        return x
