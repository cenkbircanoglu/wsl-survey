from torchvision.models import resnet

from wsl_survey.segmentation.irn.net import cam


class CAM(cam.CAM):
    def __init__(self, *args, **kwargs):
        backbone_model = resnet.resnet18(pretrained=True)
        channels = 512
        super(CAM, self).__init__(backbone_model=backbone_model,
                                  channels=channels, **kwargs)

