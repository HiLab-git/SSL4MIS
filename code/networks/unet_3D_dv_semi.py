"""
This file is adapted from https://github.com/ozan-oktay/Attention-Gated-Networks
"""

import math
import torch
import torch.nn as nn
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT, UnetDsv3
import torch.nn.functional as F
from networks.networks_other import init_weights


class unet_3D_dv_semi(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_dv_semi, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(
            in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(
            in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(
            in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(
            in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        self.dropout1 = nn.Dropout3d(p=0.5)
        self.dropout2 = nn.Dropout3d(p=0.3)
        self.dropout3 = nn.Dropout3d(p=0.2)
        self.dropout4 = nn.Dropout3d(p=0.1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up4 = self.dropout1(up4)

        up3 = self.up_concat3(conv3, up4)
        up3 = self.dropout2(up3)

        up2 = self.up_concat2(conv2, up3)
        up2 = self.dropout3(up2)

        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout4(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        return dsv1, dsv2, dsv3, dsv4

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
