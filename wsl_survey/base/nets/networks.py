import torch.nn as nn
from torchvision import models

from wsl_survey.base.nets.check_network_inputs import check_network_input_size
from wsl_survey.base.nets.network_features import network_features


class Net(nn.Module):
    def __init__(self, backbone=None, classifier=None, finetune=False, **kwargs):
        super(Net, self).__init__()

        self.backbone_model = backbone
        self.classifier = classifier
        self.finetune = finetune
        if finetune:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone_model.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        parameters = []
        for p in self.backbone_model.parameters():
            if p.requires_grad:
                parameters.append(p)
        for p in self.classifier.parameters():
            if p.requires_grad:
                parameters.append(p)
        return parameters

    def forward(self, x):
        x = self.backbone_model(x)
        x = self.classifier(x)
        return x


def load_network(version, pretrained=False, num_classes=None, classifier=None, finetune=False, return_backbone=False,
                 image_size=None):
    if image_size:
        check_network_input_size(image_size, version)
    model = getattr(models, version)
    if num_classes == 1000 or not pretrained:
        return model(pretrained=pretrained, num_classes=num_classes)
    else:
        net = model(pretrained=pretrained)
        backbone, feature_size = network_features(net, version)
        if return_backbone:
            return backbone, feature_size
        if not classifier:
            classifier = nn.Linear(feature_size, num_classes)
        return Net(backbone=backbone, classifier=classifier, finetune=finetune)
