import torch.nn as nn
import torch


class Resnet18(nn.Module):
    """Some Information about Resnet18"""

    def __init__(self, gray_scale=False):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        if gray_scale:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.layer2 = BuildingBlock(64, 64, False)
        self.layer3 = BuildingBlock(64, 64, False)
        self.layer4 = BuildingBlock(64, 128)
        self.layer5 = BuildingBlock(128, 128, False)
        self.layer6 = BuildingBlock(128, 256)
        self.layer7 = BuildingBlock(256, 256, False)
        self.layer8 = BuildingBlock(256, 512)
        self.layer9 = BuildingBlock(512, 512, False)
        self.fc = nn.Linear(512, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        # x = self.max_1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = nn.AvgPool2d(x.shape[-1])(x).reshape(-1, 512)
        x = self.fc(x)
        return x


class BuildingBlock(nn.Module):
    """Some Information about BuildingBlock"""

    def __init__(self, in_channel, out_dim, down_sample=True):
        super(BuildingBlock, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channel, out_dim, kernel_size=3, padding=1, bias=False
        )
        self.down_sample = down_sample
        if down_sample:
            self.first_conv = nn.Conv2d(
                in_channel, out_dim, kernel_size=3, padding=1, stride=2, bias=False
            )
        self.building_block = nn.Sequential(
            self.first_conv,
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.short_cut_conv = nn.Conv2d(
            in_channel, out_dim, kernel_size=1, stride=2, bias=False
        )
        self.short_cut_bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        residual = torch.clone(x)
        x = self.building_block(x)
        if self.down_sample:
            residual = self.short_cut_conv(residual)
            residual = self.short_cut_bn(residual)
        x = x + residual
        x = nn.ReLU()(x)
        return x
