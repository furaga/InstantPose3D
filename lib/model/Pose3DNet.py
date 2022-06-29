import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from abc import ABCMeta, abstractmethod

hg_layer_nums = [24 * 4 * 28]


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias
    )


class ResNetBackBone(nn.Module):
    def __init__(self, resnet_type="resnet18", pretrained=True):
        super().__init__()
        if resnet_type == "resnet18":
            self.resnet_model = torchvision.models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            self.resnet_model = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet_model = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            self.resnet_model = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            self.resnet_model = torchvision.models.resnet152(pretrained=pretrained)

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        out1 = self.resnet_model.layer1(x)  # width, heightは1/4
        out2 = self.resnet_model.layer2(out1)  # width, heightは1/8
        out3 = self.resnet_model.layer3(out2)  # width, heightは1/16
        out4 = self.resnet_model.layer4(out3)  # width, heightは1/32
        return out1, out2, out3, out4


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm="batch"):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == "batch":
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == "group":
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.conv1(F.relu(self.bn1(x), True))
        out2 = self.conv2(F.relu(self.bn2(out1), True))
        out3 = self.conv3(F.relu(self.bn3(out2), True))

        out3 = torch.cat([out1, out2, out3], 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, depth, in_ch, out_ch, norm="batch"):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module(
            "b1_" + str(level), ConvBlock(self.in_ch, self.out_ch, norm=self.norm)
        )
        self.add_module(
            "b2_" + str(level), ConvBlock(self.in_ch, self.in_ch, norm=self.norm)
        )

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module(
                "b2_plus_" + str(level),
                ConvBlock(self.in_ch, self.out_ch, norm=self.norm),
            )

        self.add_module(
            "b3_" + str(level), ConvBlock(self.out_ch, self.out_ch, norm=self.norm)
        )

    def _forward(self, level, inp):
        # upper branch
        up1 = inp
        up1 = self._modules["b1_" + str(level)](up1)

        # lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules["b2_" + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules["b2_plus_" + str(level)](low2)

        low3 = low2
        low3 = self._modules["b3_" + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode="bicubic", align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='bilinear')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGNet(nn.Module):
    def __init__(self, args):
        super(HGNet, self).__init__()

        self.resnet = ResNetBackBone(resnet_type="resnet34")
        self.args = args

        in_ch = 256 * 3
        self.n_stack = 0
        for i, out_ch in enumerate(hg_layer_nums):
            self.add_module(f"m{i}", HourGlass(self.args.hg_depth, in_ch, out_ch))
            self.add_module(
                "al" + str(i),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            )
            in_ch = out_ch
            self.n_stack += 1

    # images: B x 9 x 448 x 448
    def forward(
        self,
        images,
    ):
        # Bx9x448x448 -> Bx9x28x28
        out_resnet = [
            self.resnet.forward(images[:, 3 * i : 3 * (i + 1)])[2] for i in range(3)
        ]

        # Bx9x28x28 -> Bx28x28x28
        previous = torch.cat(out_resnet, dim=1)
        outputs = []
        for i in range(self.n_stack):
            hg = self._modules["m" + str(i)](previous)
            outputs.append(hg)
            if i < self.n_stack - 1:
                previous = self._modules["al" + str(i)](previous)
                previous = previous + hg

        # list of Bx28x28x28
        return outputs


class Pose3DNet(nn.Module):
    def __init__(self, args):
        super(Pose3DNet, self).__init__()

        self.add_module("m0", HGNet(args))

    # images: B x 9 x 448 x 448
    def forward(
        self,
        images,
    ):
        n_stack = len(hg_layer_nums)
        # -> B x (24 x 4 x ch) x 28 x 28
        outputs = self._modules["m0"](images)

        # -> B x 24 x 4 x ch x 28 x 28
        outputs = [
            o.reshape(
                (
                    o.shape[0],
                    24,
                    4,
                    -1,
                    o.shape[2],
                    o.shape[3],
                )
            )
            for o in outputs
        ]
        #        outputs = [torch.transpose(o, 0, 1, 2, 3, 4, 5) for o in outputs]
        outputs = [torch.transpose(o, 2, 3) for o in outputs]
        outputs = [torch.transpose(o, 3, 4) for o in outputs]
        outputs = [torch.transpose(o, 4, 5) for o in outputs]

        heatmaps = [o[:, :, :, :, :, 0] for o in outputs]
        offsets = [o[:, :, :, :, :, 1:] for o in outputs]
        return heatmaps, offsets
