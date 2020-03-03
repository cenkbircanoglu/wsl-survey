import torch
import torch.nn as nn
from torchvision import models

from wsl_survey.base.nets.check_network_inputs import check_network_input_size
from wsl_survey.base.nets.network_features import network_features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiOutputNet(nn.Module):
    def __init__(self, backbone=None, classifier_dict=None, finetune=False, **kwargs):
        super(MultiOutputNet, self).__init__()

        self.backbone_model = backbone
        self.classifier_dict = {key: value.to(device) for key, value in classifier_dict.items()}
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
        for classifier in self.classifier_dict.values():
            for p in classifier.parameters():
                if p.requires_grad:
                    parameters.append(p)
        return parameters

    def forward(self, x):
        x = self.backbone_model(x)
        x_dict = {}
        for key, classifier in self.classifier_dict.items():
            x_dict[key] = classifier(x)
        return x_dict


def load_multi_output_network(version, pretrained=False, feature_name_size={}, finetune=False, image_size=None):
    if image_size:
        check_network_input_size(image_size, version)
    model = getattr(models, version)
    net = model(pretrained=pretrained)
    backbone, feature_size = network_features(net, version)
    classifier_dict = {}
    for feature_name, num_classes in feature_name_size.items():
        classifier_dict[feature_name] = nn.Linear(feature_size, num_classes)
    return MultiOutputNet(backbone=backbone, classifier_dict=classifier_dict, finetune=finetune)
