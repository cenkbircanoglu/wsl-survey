import torch.nn as nn
import torch.nn.functional as F

from wsl_survey.segmentation.irn_unique.misc import torchutils


class Net(nn.Module):
    def __init__(self, backbone=None, num_classes=10582, conv_output=2048):
        super(Net, self).__init__()

        self.backbone_model = backbone
        self.num_classes = num_classes

        self.stage1 = nn.Sequential(self.backbone_model.conv1,
                                    self.backbone_model.bn1,
                                    self.backbone_model.relu,
                                    self.backbone_model.maxpool,
                                    self.backbone_model.layer1)
        self.stage2 = nn.Sequential(self.backbone_model.layer2)
        self.stage3 = nn.Sequential(self.backbone_model.layer3)
        self.stage4 = nn.Sequential(self.backbone_model.layer4)

        self.classifier = nn.Conv2d(conv_output,
                                    self.num_classes,
                                    1,
                                    bias=False)

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
        x = x.view(-1, self.num_classes)

        return x

    def train(self, mode=True):
        for p in self.backbone_model.conv1.parameters():
            p.requires_grad = False
        for p in self.backbone_model.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()),
                list(self.newly_added.parameters()))


class CAM(Net):
    def __init__(self, num_classes=10582):
        super(CAM, self).__init__(num_classes=num_classes)

    def forward(self, x):
        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x
