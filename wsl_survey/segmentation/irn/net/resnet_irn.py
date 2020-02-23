from wsl_survey.segmentation.irn.net import irn
from wsl_survey.segmentation.irn.net.resnet import resnet18, resnet34, \
    resnet50, resnet101, resnet152, deeplabv3


class ResNet18(irn.SmallNet):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge_out, dp_out = ResNet18()(x)
    >>> assert edge_out.shape == torch.Size([3, 1, 128, 128])
    >>> assert dp_out.shape == torch.Size([3, 2, 128, 128])
    """
    def __init__(self):
        backbone = resnet18(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet18, self).__init__(backbone=backbone)


class ResNet18AffinityDisplacementLoss(irn.SmallAffinityDisplacementLoss):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = ResNet18AffinityDisplacementLoss(path_index)(x, True)
    >>> assert pos_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert neg_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert dp_fg_loss.shape == torch.Size([3, 2, 152, 13090])
    >>> assert dp_bg_loss.shape == torch.Size([3, 2, 152, 13090])
    """
    def __init__(self, path_index):
        backbone = resnet18(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet18AffinityDisplacementLoss,
              self).__init__(path_index, backbone=backbone)


class ResNet18EdgeDisplacement(irn.SmallEdgeDisplacement):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge, dp = ResNet18EdgeDisplacement()(x)
    >>> assert edge.shape == torch.Size([1, 128, 128])
    >>> assert dp.shape == torch.Size([2, 128, 128])
    """
    def __init__(self):
        backbone = resnet18(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet18EdgeDisplacement, self).__init__(backbone=backbone)


class ResNet34(irn.SmallNet):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge_out, dp_out = ResNet34()(x)
    >>> assert edge_out.shape == torch.Size([3, 1, 128, 128])
    >>> assert dp_out.shape == torch.Size([3, 2, 128, 128])
    """
    def __init__(self):
        backbone = resnet34(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet34, self).__init__(backbone=backbone)


class ResNet34AffinityDisplacementLoss(irn.SmallAffinityDisplacementLoss):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = ResNet34AffinityDisplacementLoss(path_index)(x, True)
    >>> assert pos_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert neg_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert dp_fg_loss.shape == torch.Size([3, 2, 152, 13090])
    >>> assert dp_bg_loss.shape == torch.Size([3, 2, 152, 13090])
    """
    def __init__(self, path_index):
        backbone = resnet34(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet34AffinityDisplacementLoss,
              self).__init__(path_index, backbone=backbone)


class ResNet34EdgeDisplacement(irn.SmallEdgeDisplacement):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge, dp = ResNet34EdgeDisplacement()(x)
    >>> assert edge.shape == torch.Size([1, 128, 128])
    >>> assert dp.shape == torch.Size([2, 128, 128])
    """
    def __init__(self):
        backbone = resnet34(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet34EdgeDisplacement, self).__init__(backbone=backbone)


class ResNet50(irn.Net):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge_out, dp_out = ResNet50()(x)
    >>> assert edge_out.shape == torch.Size([3, 1, 128, 128])
    >>> assert dp_out.shape == torch.Size([3, 2, 128, 128])
    """
    def __init__(self):
        backbone = resnet50(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet50, self).__init__(backbone=backbone)


class ResNet50AffinityDisplacementLoss(irn.AffinityDisplacementLoss):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = ResNet50AffinityDisplacementLoss(path_index)(x, True)
    >>> assert pos_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert neg_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert dp_fg_loss.shape == torch.Size([3, 2, 152, 13090])
    >>> assert dp_bg_loss.shape == torch.Size([3, 2, 152, 13090])
    """
    def __init__(self, path_index):
        backbone = resnet50(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet50AffinityDisplacementLoss,
              self).__init__(path_index, backbone=backbone)


class ResNet50EdgeDisplacement(irn.EdgeDisplacement):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge, dp = ResNet50EdgeDisplacement()(x)
    >>> assert edge.shape == torch.Size([1, 128, 128])
    >>> assert dp.shape == torch.Size([2, 128, 128])
    """
    def __init__(self):
        backbone = resnet50(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet50EdgeDisplacement, self).__init__(backbone=backbone)


class ResNet101(irn.Net):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge_out, dp_out = ResNet101()(x)
    >>> assert edge_out.shape == torch.Size([3, 1, 128, 128])
    >>> assert dp_out.shape == torch.Size([3, 2, 128, 128])
    """
    def __init__(self):
        backbone = resnet101(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet101, self).__init__(backbone=backbone)


class ResNet101AffinityDisplacementLoss(irn.AffinityDisplacementLoss):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = ResNet101AffinityDisplacementLoss(path_index)(x, True)
    >>> assert pos_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert neg_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert dp_fg_loss.shape == torch.Size([3, 2, 152, 13090])
    >>> assert dp_bg_loss.shape == torch.Size([3, 2, 152, 13090])
    """
    def __init__(self, path_index):
        backbone = resnet101(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet101AffinityDisplacementLoss,
              self).__init__(path_index, backbone=backbone)


class ResNet101EdgeDisplacement(irn.EdgeDisplacement):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge, dp = ResNet101EdgeDisplacement()(x)
    >>> assert edge.shape == torch.Size([1, 128, 128])
    >>> assert dp.shape == torch.Size([2, 128, 128])
    """
    def __init__(self):
        backbone = resnet101(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet101EdgeDisplacement, self).__init__(backbone=backbone)


class ResNet152(irn.Net):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge_out, dp_out = ResNet152()(x)
    >>> assert edge_out.shape == torch.Size([3, 1, 128, 128])
    >>> assert dp_out.shape == torch.Size([3, 2, 128, 128])
    """
    def __init__(self):
        backbone = resnet152(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet152, self).__init__(backbone=backbone)


class ResNet152AffinityDisplacementLoss(irn.AffinityDisplacementLoss):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = ResNet152AffinityDisplacementLoss(path_index)(x, True)
    >>> assert pos_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert neg_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert dp_fg_loss.shape == torch.Size([3, 2, 152, 13090])
    >>> assert dp_bg_loss.shape == torch.Size([3, 2, 152, 13090])
    """
    def __init__(self, path_index):
        backbone = resnet152(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet152AffinityDisplacementLoss,
              self).__init__(path_index, backbone=backbone)


class ResNet152EdgeDisplacement(irn.EdgeDisplacement):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge, dp = ResNet152EdgeDisplacement()(x)
    >>> assert edge.shape == torch.Size([1, 128, 128])
    >>> assert dp.shape == torch.Size([2, 128, 128])
    """
    def __init__(self):
        backbone = resnet152(pretrained=True, strides=(2, 2, 2, 1))
        super(ResNet152EdgeDisplacement, self).__init__(backbone=backbone)


class DeepLabV3(irn.Net):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge_out, dp_out = ResNet152()(x)
    >>> assert edge_out.shape == torch.Size([3, 1, 128, 128])
    >>> assert dp_out.shape == torch.Size([3, 2, 128, 128])
    """
    def __init__(self):
        backbone = deeplabv3(pretrained=True)
        super(DeepLabV3, self).__init__(backbone=backbone)


class DeepLabV3AffinityDisplacementLoss(irn.AffinityDisplacementLoss):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = DeepLabV3AffinityDisplacementLoss(path_index)(x, True)
    >>> assert pos_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert neg_aff_loss.shape == torch.Size([3, 152, 13090])
    >>> assert dp_fg_loss.shape == torch.Size([3, 2, 152, 13090])
    >>> assert dp_bg_loss.shape == torch.Size([3, 2, 152, 13090])
    """
    def __init__(self, path_index):
        backbone = deeplabv3(pretrained=True)
        super(DeepLabV3AffinityDisplacementLoss,
              self).__init__(path_index, backbone=backbone)


class DeepLabV3EdgeDisplacement(irn.EdgeDisplacement):
    """
    >>> import torch
    >>> from wsl_survey.segmentation.irn.misc import indexing
    >>> path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    >>> x = torch.randn((3, 3, 512, 512))
    >>> edge, dp = DeepLabV3EdgeDisplacement()(x)
    >>> assert edge.shape == torch.Size([1, 128, 128])
    >>> assert dp.shape == torch.Size([2, 128, 128])
    """
    def __init__(self):
        backbone = deeplabv3(pretrained=True)
        super(DeepLabV3EdgeDisplacement, self).__init__(backbone=backbone)

if __name__ == '__main__':
    import doctest

    doctest.testmod()
