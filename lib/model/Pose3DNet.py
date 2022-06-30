from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from abc import ABCMeta, abstractmethod

# TODO: TDPT(24)に合わせたいhttps://digital-standard.com/tdpt/
K = 23  # 関節数

# hg_layer_nums[i]はすべてK * 4 * 28の約数である必要がある
hg_layer_nums = [K * 4 * 28]


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


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self, in_channels: int, atrous_rates: List[int], out_channels: int = 256
    ) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


# https://github.com/pytorch/vision/blob/87cde716b7f108f3db7b86047596ebfad1b88380/torchvision/models/segmentation/deeplabv3.py#L13
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            #            ASPP(in_channels, [12, 24, 36]),
            ASPP(in_channels, [1, 2, 4]),  # 適当
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class HGNet(nn.Module):
    def __init__(self, args):
        super(HGNet, self).__init__()
        self.args = args
        self.backbone = ResNetBackBone(
            resnet_type="resnet34", pretrained=args.pretrained_path is None
        )
        self.model = DeepLabHead(3 * 256, K * 28 * 4)

    # images: B x 9 x 448 x 448
    def forward(
        self,
        images,
    ):
        # Bx9x448x448 -> Bx9x28x28
        xs = [
            self.backbone.forward(images[:, 3 * i : 3 * (i + 1)])[2] for i in range(3)
        ]
        x = torch.cat(xs, dim=1)

        # Bx(Kx28x4)x28x28
        outputs = self.model(x)
        return outputs


class Pose3DNet(nn.Module):
    def __init__(self, args, is_train=True):
        super(Pose3DNet, self).__init__()
        self.hgnet = HGNet(args)
        self.is_train = is_train

    # images: 3 images of B x 3 x 448 x 448
    def forward(
        self,
        images,
    ):
        outputs = self.hgnet(images)

        # Bx(Kx28x4)x28x28 -> BxKx28x28x28x4
        outputs = outputs.reshape(
            (
                outputs.shape[0],
                K,
                4,
                28,
                outputs.shape[2],
                outputs.shape[3],
            )
        )
        outputs = torch.transpose(outputs, 2, 3)
        outputs = torch.transpose(outputs, 3, 4)
        outputs = torch.transpose(outputs, 4, 5)

        # BxKx28x28x28 (binary classification)
        heatmaps = outputs[:, :, :, :, :, 0]
        if not self.is_train:
            # 訓練時はBCEWithLogitsLossを使う場合のでsigmoidに通さなくて良い？
            heatmaps = torch.sigmoid(heatmaps)

        # BxKx28x28x28x3 (regression)
        offsets = outputs[:, :, :, :, :, 1:]

        return heatmaps, offsets
