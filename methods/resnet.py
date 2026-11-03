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
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), )

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

        self.relu = nn.ReLU(inplace=True)
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
        self.relu = nn.ReLU(inplace=True)
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
