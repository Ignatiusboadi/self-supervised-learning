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


