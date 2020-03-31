import unittest

import torch

from wsl_survey.base.nets.networks import load_network


class TestNetworks(unittest.TestCase):
    NETWORKS = ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'mnasnet0_5', 'mnasnet1_0',
                'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d',
                'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2']

    def test_load_network_case1(self):
        img_size = 128
        batch_size = 2

        for version in self.NETWORKS:
            x = torch.randn((batch_size, 3, img_size, img_size))
            y = load_network(version, num_classes=1000)(x)
            assert y.shape == torch.Size([batch_size, 1000])

    def test_load_network_case2(self):
        img_size = 128
        batch_size = 2

        for version in self.NETWORKS:
            x = torch.randn((batch_size, 3, img_size, img_size))
            y = load_network(version, num_classes=10)(x)
            assert y.shape == torch.Size([batch_size, 10])

    def test_load_network_case3(self):
        img_size = 128
        batch_size = 2

        for version in self.NETWORKS:
            x = torch.randn((batch_size, 3, img_size, img_size))
            y = load_network(version, pretrained=True, num_classes=1000)(x)
            assert y.shape == torch.Size([batch_size, 1000])

    def test_load_network_case4(self):
        img_size = 128
        batch_size = 2
        for version in self.NETWORKS:
            x = torch.randn((batch_size, 3, img_size, img_size))
            y = load_network(version, pretrained=True, num_classes=10)(x)
            assert y.shape == torch.Size([batch_size, 10])

    def test_load_network_case5(self):
        img_size = 128
        batch_size = 2
        for version in self.NETWORKS:
            x = torch.randn((batch_size, 3, img_size, img_size))
            network = load_network(version, pretrained=True, num_classes=28)
            y = network(x)
            assert y.shape == torch.Size([batch_size, 28])
            assert len(list(network.parameters())) == len(
                list([p for p in network.parameters() if p.requires_grad == True]))

    def test_load_network_case6(self):
        img_size = 128
        batch_size = 2
        for version in self.NETWORKS:
            x = torch.randn((batch_size, 3, img_size, img_size))
            network = load_network(version, pretrained=True, num_classes=28, finetune=True)
            y = network(x)
            assert len(list(network.parameters())) != len(
                list([p for p in network.parameters() if p.requires_grad == True]))
            assert y.shape == torch.Size([batch_size, 28])


if __name__ == '__main__':
    unittest.main()
