import torch
import torch.nn as nn
from torchviz import make_dot

from wsl_survey.segmentation.irn.net.resnet import resnet152


class OriginalNet(nn.Module):
    def __init__(self):
        super(OriginalNet, self).__init__()

        # backbone
        self.backbone_model = resnet152(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.backbone_model.conv1,
                                    self.backbone_model.bn1,
                                    self.backbone_model.relu,
                                    self.backbone_model.maxpool)
        self.stage2 = nn.Sequential(self.backbone_model.layer1)
        self.stage3 = nn.Sequential(self.backbone_model.layer2)
        self.stage4 = nn.Sequential(self.backbone_model.layer3)
        self.stage5 = nn.Sequential(self.backbone_model.layer4)
        self.mean_shift = OriginalNet.MeanShift(2)

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        # branch: displacement field
        self.fc_dp1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.fc_dp2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_dp3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_dp4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp6 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp7 = nn.Sequential(nn.Conv2d(448, 256, 1, bias=False),
                                    nn.GroupNorm(16,
                                                 256), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 2, 1, bias=False),
                                    self.mean_shift)

        self.backbone = nn.ModuleList(
            [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([
            self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4,
            self.fc_edge5, self.fc_edge6
        ])
        self.dp_layers = nn.ModuleList([
            self.fc_dp1, self.fc_dp2, self.fc_dp3, self.fc_dp4, self.fc_dp5,
            self.fc_dp6, self.fc_dp7
        ])

    class MeanShift(nn.Module):
        def __init__(self, num_features):
            super(OriginalNet.MeanShift, self).__init__()
            self.register_buffer('running_mean', torch.zeros(num_features))

        def forward(self, input):
            if self.training:
                return input
            return input - self.running_mean.view(1, 2, 1, 1)

    def forward(self, x):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_out = self.fc_edge6(
            torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        dp1 = self.fc_dp1(x1)
        dp2 = self.fc_dp2(x2)
        dp3 = self.fc_dp3(x3)
        dp4 = self.fc_dp4(x4)[..., :dp3.size(2), :dp3.size(3)]
        dp5 = self.fc_dp5(x5)[..., :dp3.size(2), :dp3.size(3)]

        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5],
                                       dim=1))[..., :dp2.size(2), :dp2.size(3)]
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))

        return edge_out, dp_out

    def trainable_parameters(self):
        return (tuple(self.edge_layers.parameters()),
                tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # backbone
        self.backbone_model = resnet152(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.backbone_model.conv1,
                                    self.backbone_model.bn1,
                                    self.backbone_model.relu,
                                    self.backbone_model.maxpool)
        self.stage2 = nn.Sequential(self.backbone_model.layer1)
        self.stage3 = nn.Sequential(self.backbone_model.layer2)
        self.stage4 = nn.Sequential(self.backbone_model.layer3)
        self.mean_shift = OriginalNet.MeanShift(2)

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )

        self.fc_edge6 = nn.Conv2d(128, 1, 1, bias=True)

        # branch: displacement field
        self.fc_dp1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.fc_dp2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_dp3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_dp4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )

        self.fc_dp6 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp7 = nn.Sequential(nn.Conv2d(448, 256, 1, bias=False),
                                    nn.GroupNorm(16,
                                                 256), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 2, 1, bias=False),
                                    self.mean_shift)

        self.backbone = nn.ModuleList(
            [self.stage1, self.stage2, self.stage3, self.stage4])
        self.edge_layers = nn.ModuleList([
            self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4,
            self.fc_edge6
        ])
        self.dp_layers = nn.ModuleList([
            self.fc_dp1, self.fc_dp2, self.fc_dp3, self.fc_dp4, self.fc_dp6,
            self.fc_dp7
        ])

    class MeanShift(nn.Module):
        def __init__(self, num_features):
            super(MyNet.MeanShift, self).__init__()
            self.register_buffer('running_mean', torch.zeros(num_features))

        def forward(self, input):
            if self.training:
                return input
            return input - self.running_mean.view(1, 2, 1, 1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4],
                                           dim=1))

        dp1 = self.fc_dp1(x1)
        dp2 = self.fc_dp2(x2)
        dp3 = self.fc_dp3(x3)
        dp4 = self.fc_dp4(x4)[..., :dp3.size(2), :dp3.size(3)]

        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4],
                                       dim=1))[..., :dp2.size(2), :dp2.size(3)]
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))

        return edge_out, dp_out

    def trainable_parameters(self):
        return (tuple(self.edge_layers.parameters()),
                tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


if __name__ == '__main__':
    # import torch

    from torchsummary import summary

    x = torch.randn((3, 3, 512, 512))
    mynet = MyNet()

    y = mynet(x)
    print(mynet)
    print(dict(mynet.named_parameters()).keys())
    dot = make_dot(y[0].mean(), params=dict(mynet.named_parameters()))
    dot.format = 'jpg'
    dot.render('mynet1')
    dot = make_dot(y[1].mean(), params=dict(mynet.named_parameters()))
    dot.format = 'jpg'
    dot.render('mynet2')

    orgnet = OriginalNet()
    print(orgnet)
    print(dict(orgnet.named_parameters()).keys())
    y = orgnet(x)
    dot = make_dot(y[0].mean(), params=dict(orgnet.named_parameters()))
    dot.format = 'jpg'
    dot.render('orgnet1')
    dot = make_dot(y[1].mean(), params=dict(orgnet.named_parameters()))
    dot.format = 'jpg'
    dot.render('orgnet2')

    summary(mynet, input_size=(3, 512, 512))

    summary(orgnet, input_size=(3, 512, 512))
