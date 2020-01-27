from wsl_survey.segmentation.irn.net.distilled import cam
from wsl_survey.segmentation.irn.net.resnet import resnet50, resnet152, \
    resnet101


class ResNet50(cam.Net):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet50()(x)
    >>> assert y.shape == torch.Size([3, 20])
    >>> sum(p.numel() for p in ResNet50().parameters())
    23528512
    """

    def __init__(self, num_classes=20):
        backbone = resnet50(pretrained=True, strides=(2, 2, 2, 1))
        conv_output = 1024
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
    23528512
    """


class ResNet101(cam.Net):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet101()(x)
    >>> assert y.shape == torch.Size([3, 20])
    >>> sum(p.numel() for p in ResNet101().parameters())
    42520640
    """

    def __init__(self, num_classes=20):
        backbone = resnet101(pretrained=True, strides=(2, 2, 2, 1))
        conv_output = 1024
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
    42520640
    """


class ResNet152(cam.Net):
    """
    >>> import torch
    >>> x = torch.randn((3, 3, 512, 512))
    >>> y = ResNet152()(x)
    >>> assert y.shape == torch.Size([3, 20])
    >>> sum(p.numel() for p in ResNet152().parameters())
    58164288
    """

    def __init__(self, num_classes=20):
        backbone = resnet152(pretrained=True, strides=(2, 2, 2, 1))
        conv_output = 1024
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
    58164288
    """


if __name__ == '__main__':
    import doctest

    doctest.testmod()
