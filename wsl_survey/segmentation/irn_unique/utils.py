from functools import partial

from wsl_survey.segmentation.irn_unique.misc import indexing
from wsl_survey.segmentation.irn_unique.net import resnet_cam, resnet_irn


def find_smallest(nets):
    smallest_value = float('inf')
    smallest_net = None
    for net in nets:
        total_param = sum(p.numel() for p in net().parameters())
        if smallest_value > total_param:
            smallest_value = total_param
            smallest_net = net
    print(smallest_net)
    return smallest_net


if __name__ == '__main__':
    nets = [
        resnet_cam.ResNet18, resnet_cam.ResNet34, resnet_cam.ResNet50,
        resnet_cam.ResNet101, resnet_cam.ResNet152
    ]
    find_smallest(nets)

    nets = [
        resnet_cam.ResNet18CAM, resnet_cam.ResNet34CAM, resnet_cam.ResNet50CAM,
        resnet_cam.ResNet101CAM, resnet_cam.ResNet152CAM
    ]
    find_smallest(nets)

    nets = [
        resnet_irn_unique.ResNet18, resnet_irn_unique.ResNet34, resnet_irn_unique.ResNet50,
        resnet_irn_unique.ResNet101, resnet_irn_unique.ResNet152
    ]
    find_smallest(nets)

    path_index = indexing.PathIndex(radius=10,
                                    default_size=(512 // 4, 512 // 4))

    nets = [
        partial(resnet_irn_unique.ResNet18AffinityDisplacementLoss,
                path_index=path_index),
        partial(resnet_irn_unique.ResNet34AffinityDisplacementLoss,
                path_index=path_index),
        partial(resnet_irn_unique.ResNet50AffinityDisplacementLoss,
                path_index=path_index),
        partial(resnet_irn_unique.ResNet101AffinityDisplacementLoss,
                path_index=path_index),
        partial(resnet_irn_unique.ResNet152AffinityDisplacementLoss,
                path_index=path_index)
    ]
    find_smallest(nets)

    nets = [
        resnet_irn_unique.ResNet18EdgeDisplacement,
        resnet_irn_unique.ResNet34EdgeDisplacement,
        resnet_irn_unique.ResNet50EdgeDisplacement,
        resnet_irn_unique.ResNet101EdgeDisplacement,
        resnet_irn_unique.ResNet152EdgeDisplacement
    ]
    find_smallest(nets)
