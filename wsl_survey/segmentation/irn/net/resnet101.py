import torch.utils.model_zoo as model_zoo

from wsl_survey.segmentation.irn.net.resnet import Bottleneck, ResNet, \
    model_urls


def resnet101(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model
