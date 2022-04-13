from ast import arg
from operator import xor
from os import device_encoding
from tokenize import Special
import numpy as np
import torch
import torch.nn as nn
import itertools
import pandas

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform(layer.weight)


class PointNet(nn.Module):
    """Some Information about PointNet"""

    def __init__(self, point_size, num_classes):
        super(PointNet, self).__init__()
        self.t_net1 = T_Net(point_size, 3)
        self.t_net2 = T_Net(point_size, 64)
        self.geom_mlp1 = Special_shared_layer(3, 64)
        self.geom_mlp2 = Special_shared_layer(64, 64)
        self.feat_mlp1 = Special_shared_layer(64, 64)
        self.feat_mlp2 = Special_shared_layer(64, 128)
        self.feat_mlp3 = Special_shared_layer(128, 1024)
        self.max_pool = nn.MaxPool2d(kernel_size=(point_size, 1))
        self.head = Segmentation_head(num_classes)
        # first iteration T net transformation is identity, enfoce this using
        # a flag. Set to false after the first iteration is over
        self.first_pass = True

    def forward(self, x):
        x = x.unsqueeze(1)
        transform = self.t_net1(x)
        x = torch.bmm(x.squeeze(1), transform.squeeze(1)).unsqueeze(1)
        x = self.geom_mlp1(x)
        x = self.geom_mlp2(x)
        transform = self.t_net2(x)
        x = torch.bmm(x.squeeze(1), transform.squeeze(1)).unsqueeze(1)
        local_feat = torch.clone(x)
        x = self.feat_mlp1(x)
        x = self.feat_mlp2(x)
        x = self.feat_mlp3(x)
        x = self.max_pool(x)
        x = x.reshape((x.size(0), x.size(-1)))
        x = x.unsqueeze(1).repeat(1, local_feat.shape[-2], 1).unsqueeze(1)
        x = torch.cat([local_feat, x], dim=-1)
        x = self.head(x).squeeze(1)
        return x


class Segmentation_head(nn.Module):
    """Some Information about Segmentation_head"""

    def __init__(self, num_classes):
        super(Segmentation_head, self).__init__()
        self.segment_mlp1 = Special_shared_layer(1088, 512)
        self.segment_mlp2 = Special_shared_layer(512, 256)
        self.segment_mlp3 = Special_shared_layer(256, 128)
        self.segment_mlp_out = Special_last_layer(128, num_classes)

    def forward(self, x):
        x = self.segment_mlp1(x)
        x = self.segment_mlp2(x)
        x = self.segment_mlp3(x)
        x = self.segment_mlp_out(x)
        return x


class Special_last_layer(nn.Module):
    """Some Information about Special_last_layer"""

    def __init__(self, in_p_size, out_p_size):
        super(Special_last_layer, self).__init__()
        self.special_mlp = nn.Conv2d(
            in_channels=1,
            out_channels=out_p_size,
            kernel_size=(1, in_p_size),
            bias=False,
        )

    def forward(self, x):
        x = self.special_mlp(x)
        x = x.moveaxis(0, -1).T
        return x


class Special_shared_layer(nn.Module):
    """
    Implement the shared MLP layers in PointNet using Convolution's sliding.
    """

    def __init__(self, in_p_size, out_p_size):
        super(Special_shared_layer, self).__init__()
        self.special_mlp = nn.Conv2d(
            in_channels=1,
            out_channels=out_p_size,
            kernel_size=(1, in_p_size),
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.special_mlp(x)
        x = x.moveaxis(0, -1).T
        x = self.batch_norm(x)
        x = nn.ReLU(inplace=True)(x)
        return x


class T_Net(nn.Module):
    """
    T-Net with changable input and output dimensions
    """

    def __init__(self, sym_size, output_size):
        super(T_Net, self).__init__()
        self.mlp1 = Special_shared_layer(output_size, 64)
        self.mlp2 = Special_shared_layer(64, 128)
        self.mlp3 = Special_shared_layer(128, 1024)
        self.max_pool = nn.MaxPool2d(kernel_size=(sym_size, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, output_size * output_size)
        self.fc3.weight.data.fill_(0.0)
        self.fc3.bias.data.fill_(0.0)
        self.output_size = output_size

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.max_pool(x)
        x = x.reshape((x.size(0), x.size(-1)))
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc3(x)
        x = x.reshape((x.size(0), self.output_size, self.output_size))
        x += (
            torch.eye(self.output_size, device=device)
            .repeat(x.size(0), 1)
            .view((-1, self.output_size, self.output_size))
        )
        return x.unsqueeze(dim=1)
