import torch.utils.model_zoo as model_zoo

from wsl_survey.segmentation.irn.net.resnet import ResNet, model_urls
from wsl_survey.segmentation.irn.net.resnet101 import Bottleneck


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model

