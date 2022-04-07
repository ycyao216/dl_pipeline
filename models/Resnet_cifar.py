import torch.nn as nn
import torch


class Resnet18_cifar(nn.Module):
    def __init__(self):
        super(Resnet18_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.layer2 = RNC_stack(16, 16, False)
        self.layer3 = RNC_stack(16, 16, False)
        self.layer4 = RNC_stack(16, 32)
        self.layer5 = RNC_stack(32, 32, False)
        self.layer6 = RNC_stack(32, 64)
        self.layer7 = RNC_stack(64, 64, False)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.avg(x).squeeze(dim=2).squeeze(dim=2)
        x = self.fc(x)
        return x


class RNC_stack(nn.Module):
    def __init__(self, in_channel, out_dim, down_sample=True):
        super(RNC_stack, self).__init__()
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
        x = nn.ReLU()(x)
        return x
