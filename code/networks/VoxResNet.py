# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, r):
        super(SEBlock, self).__init__()

        redu_chns = int(in_channels / r)
        self.se_layers = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, redu_chns, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(redu_chns, in_channels, kernel_size=1, padding=0),
            nn.ReLU())

    def forward(self, x):
        f = self.se_layers(x)
        return f * x + x


class VoxRex(nn.Module):
    def __init__(self, in_channels):
        super(VoxRex, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)+x


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.conv_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class VoxResNet(nn.Module):
    def __init__(self, in_chns=1, feature_chns=64, class_num=2):
        super(VoxResNet, self).__init__()
        self.in_chns = in_chns
        self.ft_chns = feature_chns
        self.n_class = class_num

        self.conv1 = nn.Conv3d(in_chns, feature_chns, kernel_size=3, padding=1)
        self.res1 = VoxRex(feature_chns)
        self.res2 = VoxRex(feature_chns)
        self.res3 = VoxRex(feature_chns)
        self.res4 = VoxRex(feature_chns)
        self.res5 = VoxRex(feature_chns)
        self.res6 = VoxRex(feature_chns)

        self.up1 = UpBlock(feature_chns * 2, feature_chns)
        self.up2 = UpBlock(feature_chns * 2, feature_chns)

        self.out = nn.Conv3d(feature_chns, self.n_class, kernel_size=1)

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x2_pool = self.maxpool(x2)
        x3 = self.res3(x2_pool)
        x4 = self.maxpool(self.res4(x3))
        x5 = self.res5(x4)
        x6 = self.res6(x5)
        up1 = self.up1(x6, x2_pool)
        up2 = self.up2(up1, x)
        up = self.upsample(up2)
        out = self.out(up)
        return out
