from wsl_survey.segmentation.irn.net import cam
from wsl_survey.segmentation.irn.net.resnet import resnet18, resnet34, \
    resnet50, resnet152, resnet101


class ResNet18(cam.Net):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet18()(x)
    >>> assert y.shape == torch.Size([3, 20])
    >>> sum(p.numel() for p in ResNet18().parameters())
    11186752
    """
    def __init__(self, num_classes=2):
        backbone = resnet18(pretrained=True, strides=(2, 2, 2, 1))
        conv_output = 512
        super(ResNet18, self).__init__(backbone=backbone,
                                       num_classes=num_classes,
                                       conv_output=conv_output)


class ResNet18CAM(cam.CAM, ResNet18):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet18CAM()(x)
    >>> assert y.shape == torch.Size([20, 32, 32])
    >>> sum(p.numel() for p in ResNet18CAM().parameters())
    11186752
    """


class ResNet34(cam.Net):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet34()(x)
    >>> assert y.shape == torch.Size([3, 20])
    >>> sum(p.numel() for p in ResNet34().parameters())
    21294912
    """
    def __init__(self, num_classes=2):
        backbone = resnet34(pretrained=True, strides=(2, 2, 2, 1))
        conv_output = 512
        super(ResNet34, self).__init__(backbone=backbone,
                                       num_classes=num_classes,
                                       conv_output=conv_output)


class ResNet34CAM(cam.CAM, ResNet34):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet34CAM()(x)
    >>> assert y.shape == torch.Size([20, 32, 32])
    >>> sum(p.numel() for p in ResNet34CAM().parameters())
    21294912
    """


class ResNet50(cam.Net):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet50()(x)
    >>> assert y.shape == torch.Size([3, 20])
    >>> sum(p.numel() for p in ResNet50().parameters())
    23548992
    """
    def __init__(self, num_classes=2):
        backbone = resnet50(pretrained=True, strides=(2, 2, 2, 1))
        conv_output = 2048
        super(ResNet50, self).__init__(backbone=backbone,
                                       num_classes=num_classes,
                                       conv_output=conv_output)


class ResNet50CAM(cam.CAM, ResNet50):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet50CAM()(x)
    >>> assert y.shape == torch.Size([20, 32, 32])
    >>> sum(p.numel() for p in ResNet50CAM().parameters())
    23548992
    """


class ResNet101(cam.Net):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet101()(x)
    >>> assert y.shape == torch.Size([3, 20])
    >>> sum(p.numel() for p in ResNet101().parameters())
    42541120
    """
    def __init__(self, num_classes=2):
        backbone = resnet101(pretrained=True, strides=(2, 2, 2, 1))
        conv_output = 2048
        super(ResNet101, self).__init__(backbone=backbone,
                                        num_classes=num_classes,
                                        conv_output=conv_output)


class ResNet101CAM(cam.CAM, ResNet101):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet101CAM()(x)
    >>> assert y.shape == torch.Size([20, 32, 32])
    >>> sum(p.numel() for p in ResNet101CAM().parameters())
    42541120
    """


class ResNet152(cam.Net):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet152()(x)
    >>> assert y.shape == torch.Size([3, 20])
    >>> sum(p.numel() for p in ResNet152().parameters())
    58184768
    """
    def __init__(self, num_classes=2):
        backbone = resnet152(pretrained=True, strides=(2, 2, 2, 1))
        conv_output = 2048
        super(ResNet152, self).__init__(backbone=backbone,
                                        num_classes=num_classes,
                                        conv_output=conv_output)


class ResNet152CAM(cam.CAM, ResNet152):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet152CAM()(x)
    >>> assert y.shape == torch.Size([20, 32, 32])
    >>> sum(p.numel() for p in ResNet152CAM().parameters())
    58184768
    """


if __name__ == '__main__':
    import doctest

    doctest.testmod()
