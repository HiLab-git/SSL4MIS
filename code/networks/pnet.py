
# -*- coding: utf-8 -*-
"""
An PyTorch implementation of the DeepIGeoS paper:
     Wang, Guotai and Zuluaga, Maria A and Li, Wenqi and Pratt, Rosalind and Patel, Premal A and Aertsen, Michael and Doel, Tom and David, Anna L and Deprest, Jan and Ourselin, S{\'e}bastien and others:
     DeepIGeoS: a deep interactive geodesic framework for medical image segmentation. 
     TPAMI (7) 2018: 1559--1572
Note that there are some modifications from the original paper, such as
the use of leaky relu here.
"""
from __future__ import division, print_function

import torch
import torch.nn as nn


class PNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, padding):
        super(PNetBlock, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channels
        self.dilation = dilation
        self.padding = padding

        self.conv1 = nn.Conv2d(self.in_chns,  self.out_chns, kernel_size=3,
                               padding=self.padding, dilation=self.dilation, groups=1, bias=True)
        self.conv2 = nn.Conv2d(self.out_chns, self.out_chns, kernel_size=3,
                               padding=self.padding, dilation=self.dilation, groups=1, bias=True)
        self.in1 = nn.BatchNorm2d(self.out_chns)
        self.in2 = nn.BatchNorm2d(self.out_chns)
        self.ac1 = nn.LeakyReLU()
        self.ac2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.ac2(x)
        return x


class ConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatBlock, self).__init__()
        self.in_chns = in_channels
        self.out_chns = out_channels
        self.conv1 = nn.Conv2d(
            self.in_chns,  self.in_chns, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            self.in_chns, self.out_chns, kernel_size=1, padding=0)
        self.ac1 = nn.LeakyReLU()
        self.ac2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.ac2(x)
        return x


class OutPutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutPutBlock, self).__init__()
        self.in_chns = in_channels
        self.out_chns = out_channels
        self.conv1 = nn.Conv2d(
            self.in_chns,  self.in_chns // 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            self.in_chns // 2, self.out_chns, kernel_size=1, padding=0)
        self.drop1 = nn.Dropout2d(0.3)
        self.drop2 = nn.Dropout2d(0.3)
        self.ac1 = nn.LeakyReLU()

    def forward(self, x):
        x = self.drop1(x)
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.drop2(x)
        x = self.conv2(x)
        return x


class PNet2D(nn.Module):
    def __init__(self, in_chns, out_chns, num_filters, ratios):
        super(PNet2D, self).__init__()

        self.in_chns = in_chns
        self.out_chns = out_chns
        self.ratios = ratios
        self.num_filters = num_filters

        self.block1 = PNetBlock(
            self.in_chns, self.num_filters, self.ratios[0], padding=self.ratios[0])

        self.block2 = PNetBlock(
            self.num_filters, self.num_filters, self.ratios[1], padding=self.ratios[1])

        self.block3 = PNetBlock(
            self.num_filters, self.num_filters, self.ratios[2], padding=self.ratios[2])

        self.block4 = PNetBlock(
            self.num_filters, self.num_filters, self.ratios[3], padding=self.ratios[3])

        self.block5 = PNetBlock(
            self.num_filters, self.num_filters, self.ratios[4], padding=self.ratios[4])
        self.catblock = ConcatBlock(self.num_filters * 5, self.num_filters * 2)
        self.out = OutPutBlock(self.num_filters * 2, self.out_chns)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        conx = torch.cat([x1, x2, x3, x4, x5], dim=1)
        conx = self.catblock(conx)
        out = self.out(conx)
        return out
