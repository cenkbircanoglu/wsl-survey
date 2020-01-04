import torch

from wsl_survey.segmentation.irn.net.resnet import Bottleneck, ResNet


def tencent_resnet101(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        checkpoint = torch.load('./models/')
        state_dict = checkpoint['model']
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model
