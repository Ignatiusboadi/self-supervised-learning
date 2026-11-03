import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16, min_dim=32, ):
        super().__init__()

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, ),
            nn.BatchNorm2d(out_channels), nn.ReLU(), )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False, ),
            nn.BatchNorm2d(out_channels), nn.ReLU(), )

        attention_dim = max(out_channels // reduction, min_dim, )
        self.attention_reduce = nn.Sequential(nn.Conv2d(out_channels, attention_dim, kernel_size=1, bias=False, ),
                                              nn.ReLU(), )
        self.attention_expand = nn.Conv2d(attention_dim, 2 * out_channels, kernel_size=1, )
        self.out_channels = out_channels

    def forward(self, x):
        a = self.branch3(x)
        b = self.branch5(x)
        combined = a + b
        descriptor = F.adaptive_avg_pool2d(combined, 1)
        descriptor = self.attention_reduce(descriptor)
        attention = self.attention_expand(descriptor)
        batch_size = x.shape[0]
        attention = attention.reshape(batch_size, 2, self.out_channels, 1, 1, )
        attention = torch.softmax(attention, dim=1)
        return attention[:, 0] * a + attention[:, 1] * b


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, planes, stride=1, use_sk=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if use_sk:
            self.conv2 = SelectiveKernelConv(planes, planes, stride=1)
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes), )

        self.relu = nn.ReLU()
        out_channels = planes * self.expansion
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
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
        out = out + identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, stride=1, use_sk=False):
        super().__init__()
        width = planes
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        if use_sk:
            self.conv2 = SelectiveKernelConv(width, width, stride=stride)
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(width),
            )

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        out_channels = planes * self.expansion

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels), )
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
    BLOCK_CONFIG = {
        "resnet18": (BasicBlock, (2, 2, 2, 2)),
        "resnet34": (BasicBlock, (3, 4, 6, 3)),
        "resnet50": (Bottleneck, (3, 4, 6, 3)),
        "resnet101": (Bottleneck, (3, 4, 23, 3)),
        "resnet152": (Bottleneck, (3, 8, 36, 3)),
    }
    def __init__(self, resnet_variant="resnet18", width_multiplier=1, use_sk=False,
        in_channels=3, small_image_stem=True, zero_init_residual=False, ):
        super().__init__()
        if resnet_variant not in self.BLOCK_CONFIG:
            raise ValueError(f"Unsupported model: {resnet_variant}. Choose from {list(self.BLOCK_CONFIG)}.")
        if width_multiplier not in (1, 2, 3):
            raise ValueError("width_multiplier must be 1, 2, or 3.")
        block, layers = self.BLOCK_CONFIG[resnet_variant]
        self.resnet_variant = resnet_variant
        self.width_multiplier = width_multiplier
        self.use_sk = use_sk
        self.small_image_stem = small_image_stem
        self.in_channels = 64 * width_multiplier
        self.block = block
        stem_channels = 64 * width_multiplier

        if small_image_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
            self.maxpool = nn.Identity()
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True), )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64 * width_multiplier, layers[0], stride=1)
        self.layer2 = self._make_layer(128 * width_multiplier, layers[1], stride=2)
        self.layer3 = self._make_layer(256 * width_multiplier, layers[2], stride=2)
        self.layer4 = self._make_layer(512 * width_multiplier, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512 * width_multiplier * block.expansion
        self._initialize_weights(zero_init_residual)

    def _make_layer(self, planes, blocks, stride):
        layers = [self.block(self.in_channels, planes, stride=stride, use_sk=self.use_sk)]
        self.in_channels = planes * self.block.expansion

        for _ in range(1, blocks):
            layers.append(self.block(self.in_channels, planes, stride=1, use_sk=self.use_sk))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual=False):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.zeros_(module.bn3.weight)
                elif isinstance(module, BasicBlock):
                    if isinstance(module.conv2, SelectiveKernelConv):
                        nn.init.zeros_(module.conv2.branch3[1].weight)
                    else:
                        nn.init.zeros_(module.conv2[1].weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return torch.flatten(x, 1)
