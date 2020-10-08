import torch.nn as nn
import torch.nn.functional as F
import torch


class FC3DDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FC3DDiscriminator, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(
            n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(
            ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(
            ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((6, 6, 6))  # (D/16, W/16, H/16)
        self.classifier = nn.Linear(ndf*8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def forward(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)

        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x
