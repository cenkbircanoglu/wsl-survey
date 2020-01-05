from torchvision.models import resnet

from wsl_survey.segmentation.irn.net import cam


class CAM(cam.CAM):
    def __init__(self, *args, **kwargs):
        backbone_model = resnet.resnet101(pretrained=True)
        channels = 2048
        super(CAM, self).__init__(backbone_model=backbone_model,
                                  channels=channels, **kwargs)
