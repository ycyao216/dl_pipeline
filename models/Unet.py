import math
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Old implementation of Unet from homework 0.


class Simple_UNet(nn.Module):
    def __init__(self, num_classes):
        super(Simple_UNet, self).__init__()
        #         self.device = device
        self.d_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64, affine=False),
        )
        torch.nn.init.normal(self.d_conv1[0].weight, std=math.sqrt(2 / (27)))
        torch.nn.init.normal(self.d_conv1[2].weight, std=math.sqrt(2 / (64 * 9)))

        self.d_conv2 = self.down_conv_unit(64)
        self.d_conv3 = self.down_conv_unit(128)
        self.d_conv4 = self.down_conv_unit(256)
        self.d_conv5 = self.down_conv_unit(512)
        self.d_to_u = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.u_conv1 = self.up_conv_unit(1024)
        self.u_conv2 = self.up_conv_unit(512)
        self.u_conv3 = self.up_conv_unit(256)
        self.fina_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
        )
        torch.nn.init.normal(self.fina_conv[0].weight, std=math.sqrt(2 / (128 * 9)))
        torch.nn.init.normal(self.fina_conv[2].weight, std=math.sqrt(2 / (64 * 9)))

        # Was used when Unet's softmax and cross entropy was used.
        # self.u_conv_head = nn.Conv2d(64,2,1,padding="same")
        self.u_conv_head = nn.Conv2d(64, num_classes, 1, padding="same")
        torch.nn.init.normal(self.u_conv_head.weight, std=math.sqrt(2 / (64 * 4)))

    def down_conv_unit(self, in_channel):
        first_conv = nn.Conv2d(in_channel, in_channel * 2, 3, padding="same")
        torch.nn.init.normal(first_conv.weight, std=math.sqrt(2 / (in_channel * 9)))
        second_conv = nn.Conv2d(in_channel * 2, in_channel * 2, 3, padding="same")
        torch.nn.init.normal(
            second_conv.weight, std=math.sqrt(2 / (in_channel * 2 * 9))
        )
        first_relu = nn.ReLU()
        second_relu = nn.ReLU()
        unit = nn.Sequential(
            nn.MaxPool2d(2),
            first_conv,
            first_relu,
            second_conv,
            second_relu,
            nn.BatchNorm2d(in_channel * 2, affine=False),
        )
        return unit

    def up_conv_unit(self, in_channel):
        first_conv = nn.Conv2d(in_channel, in_channel // 2, 3, padding="same")
        torch.nn.init.normal(first_conv.weight, std=math.sqrt(2 / (in_channel * 9)))
        second_conv = nn.Conv2d(in_channel // 2, in_channel // 2, 3, padding="same")
        torch.nn.init.normal(
            first_conv.weight, std=math.sqrt(2 / (in_channel // 2 * 9))
        )
        trans_conv = nn.ConvTranspose2d(in_channel // 2, in_channel // 4, 2, stride=2)
        torch.nn.init.normal(
            trans_conv.weight, std=math.sqrt(2 / (in_channel // 2 * 4))
        )
        unit = nn.Sequential(
            first_conv,
            nn.ReLU(),
            second_conv,
            nn.ReLU(),
            nn.BatchNorm2d(in_channel // 2),
            trans_conv,
        )
        return unit

    def copy_upconv(self, x, up_conv_unit, lower_feature, in_channel):
        #         print("lower feature size", lower_feature.size())
        #         print("input size", x.size())
        if lower_feature.size() == x.size():
            x = torch.cat((lower_feature, x), dim=1)
        else:
            x = torch.cat(
                (self.copy_crop(lower_feature, torch.tensor(x.size()[2:]) * 2), x),
                dim=1,
            )
        x = up_conv_unit(x)
        return x

    def copy_crop(self, x, crop_size):
        assert x.size()[2] % 2 == 0
        assert x.size()[3] % 2 == 0
        width = x.size()[2]
        height = x.size()[3]
        center = torch.tensor((width / 2, height / 2), dtype=torch.int32)
        crop_size = torch.tensor(crop_size, dtype=torch.int32)
        #         print("center",center)
        #         print("crop_size",crop_size/2)
        begin = (center - crop_size / 2).type(torch.int32)
        #         print("begin",begin[0])
        return x[
            :, :, begin[0] : begin[0] + crop_size[0], begin[1] : begin[1] + crop_size[1]
        ]

    def forward(self, x):
        x = self.d_conv1(x)
        feat_d1 = torch.clone(x)
        x = self.d_conv2(x)
        feat_d2 = torch.clone(x)
        x = self.d_conv3(x)
        feat_d3 = torch.clone(x)
        x = self.d_conv4(x)
        feat_d4 = torch.clone(x)
        x = self.d_conv5(x)
        x = self.d_to_u(x)
        x = self.copy_upconv(x, self.u_conv1, feat_d4, x.size()[1])
        x = self.copy_upconv(x, self.u_conv2, feat_d3, x.size()[1])
        x = self.copy_upconv(x, self.u_conv3, feat_d2, x.size()[1])
        x = torch.cat((feat_d1, x), dim=1)
        x = self.fina_conv(x)
        x = self.u_conv_head(x)
        return x.float()
