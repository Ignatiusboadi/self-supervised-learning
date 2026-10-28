from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



class SelectiveKernelConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
        min_dim: int = 32,
    ):
        super().__init__()

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride,
                padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride,
                padding=2, dilation=2, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        attention_dim = max(out_channels // reduction, min_dim)

        self.attention_reduce = nn.Sequential(
            nn.Conv2d(
                out_channels, attention_dim,
                kernel_size=1, bias=False,
            ),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True),
        )

        self.attention_expand = nn.Conv2d(
            attention_dim,
            2 * out_channels,
            kernel_size=1,
        )

        self.out_channels = out_channels

    def forward(self, x):
        a = self.branch3(x)
        b = self.branch5(x)

        combined = a + b
        descriptor = F.adaptive_avg_pool2d(combined, 1)
        descriptor = self.attention_reduce(descriptor)

        batch_size = x.shape[0]
        attention = self.attention_expand(descriptor)

        attention = attention.reshape(
            batch_size, 2, self.out_channels, 1, 1
        )
        attention = torch.softmax(attention, dim=1)

        return attention[:, 0] * a + attention[:, 1] * b


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        use_sk: bool = False,
        groups: int = 1,
        base_width: int = 64,
    ):
        super().__init__()

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(
            in_channels, width, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width)

        if use_sk:
            self.conv2 = SelectiveKernelConv(
                width, width, stride=stride
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    width, width,
                    kernel_size=3, stride=stride,
                    padding=1, groups=groups, bias=False,
                ),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )

        self.bn2 = (
            nn.Identity() if use_sk else nn.Identity()
        )

        self.conv3 = nn.Conv2d(
            width, planes * self.expansion,
            kernel_size=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        out_channels = planes * self.expansion

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        if not isinstance(self.conv2, SelectiveKernelConv):
            out = self.relu(out)

        out = self.bn3(self.conv3(out))
        out = out + identity

        return self.relu(out)



class SimCLRResNetEncoder(nn.Module):
    RESNETS = {
        'resnet18': 18,
        'resnet34': 34,
        'resnet50': 50,
        'resnet101': 101,
        'resnet152': 152
    }
    DEPTHS = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }

    def __init__(
        self,
        resnet_variant: str = "resnet18",
        width_multiplier: int = 1,
        use_sk: bool = False,
        in_channels: int = 3,
        small_image_stem: bool = False,
    ):
        super().__init__()

        if resnet_variant not in self.RESNETS:
            raise ValueError(
                f"Unsupported model {resnet_variant}. "
                f"Choose from {list(self.RESNETS)}."
            )

        if width_multiplier not in (1, 2, 3):
            raise ValueError(
                "width_multiplier must be 1, 2, or 3."
            )

        self.depth = self.RESNETS[resnet_variant]
        self.width_multiplier = width_multiplier
        self.use_sk = use_sk

        stem_channels = 64 * width_multiplier

        if small_image_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels, stem_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=False,
                ),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
            self.maxpool = nn.Identity()
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels, stem_channels,
                    kernel_size=7, stride=2,
                    padding=3, bias=False,
                ),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )

        self.in_channels = stem_channels

        self.layer1 = self._make_layer(
            64 * width_multiplier, self.DEPTHS[self.depth][0], 1
        )
        self.layer2 = self._make_layer(
            128 * width_multiplier, self.DEPTHS[self.depth][1], 2
        )
        self.layer3 = self._make_layer(
            256 * width_multiplier, self.DEPTHS[self.depth][2], 2
        )
        self.layer4 = self._make_layer(
            512 * width_multiplier, self.DEPTHS[self.depth][3], 2
        )

        self.feature_dim = 2048 * width_multiplier
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride):
        layers = [
            Bottleneck(
                self.in_channels,
                planes,
                stride=stride,
                use_sk=self.use_sk,
            )
        ]

        self.in_channels = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.in_channels,
                    planes,
                    use_sk=self.use_sk,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return torch.flatten(self.avgpool(x), 1)

