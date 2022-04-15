from ast import arg
from operator import xor
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
    elif isinstance(layer, nn.Conv2d):
        torch.nn.init.kaiming_normal(layer.weight)


############################# Point net ####################################


class PointNet(nn.Module):
    """Some Information about PointNet"""

    def __init__(self, point_size, device, use_t_net=True):
        super(PointNet, self).__init__()
        self.t_net1 = T_Net(point_size, 3)
        self.t_net2 = T_Net(point_size, 64)
        self.geom_mlp1 = Special_shared_layer(3, 64)
        self.geom_mlp2 = Special_shared_layer(64, 64)
        self.feat_mlp1 = Special_shared_layer(64, 64)
        self.feat_mlp2 = Special_shared_layer(64, 128)
        self.feat_mlp3 = Special_shared_layer(128, 1024)
        self.max_pool = nn.MaxPool2d(kernel_size=(point_size, 1))
        self.head = position_transform(1024, device)
        self.use_t_net = use_t_net
        self.device = device
        # first iteration T net transformation is identity, enfoce this using
        # a flag. Set to false after the first iteration is over
        self.first_pass = True

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.use_t_net:
            if self.first_pass:
                transform = torch.eye(3).to(self.device)
                self.first_pass = False
            else:
                transform = self.t_net1(x)
            x = x @ transform
        x = self.geom_mlp1(x)
        x = self.geom_mlp2(x)
        if self.use_t_net:
            if self.first_pass:
                transform = torch.eye(64).to(self.device)
                self.first_pass = False
            else:
                transform = self.t_net2(x)
            x = x @ transform
        x = self.feat_mlp1(x)
        x = self.feat_mlp2(x)
        x = self.feat_mlp3(x)
        x = self.max_pool(x)
        x = x.reshape((x.size(0), x.size(-1)))
        if self.head is not None:
            x = self.head(x)
        return x


class position_transform(nn.Module):
    """
    The transformation predicting head
    """

    def __init__(self, in_shape, device, is_6d=False):
        super(position_transform, self).__init__()
        self.fc1 = nn.Linear(in_shape, 512)
        self.bc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 9)
        if not is_6d:
            self.fc3 = nn.Linear(256, 12)
        self.is_6d = is_6d
        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = self.bc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc2(x)
        x = self.bc2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc3(x)
        if self.is_6d:
            x = self.gram_schmidt_helper(x)
        else:
            x = self.symmetric_orthogonalization(x)
        return x

    def gram_schmidt_helper(self, in_mat):
        rot_mat = torch.zeros((in_mat.size(0), 4, 3)).to(self.device)
        rot_mat[:, :1, :] = nn.functional.normalize(in_mat[:, :3], dim=1).unsqueeze(1)
        rot_mat[:, 1:2, :] = nn.functional.normalize(in_mat[:, 3:6], dim=1).unsqueeze(1)
        rot_mat[:, 1:2, :] = (
            rot_mat[:, 1:2, :]
            - torch.sum(
                rot_mat[:, :1, :].clone() * rot_mat[:, 1:2, :].clone(), dim=-1
            ).unsqueeze(-1)
            * rot_mat[:, :1, :].clone()
        )
        rot_mat[:, 2:3, :] = torch.cross(
            rot_mat[:, :1, :].clone(), rot_mat[:, 1:2, :].clone(), dim=-1
        )
        rot_mat[:, 3:4, :] = in_mat[:, 6:].unsqueeze(1)
        return torch.transpose(rot_mat, -2, -1)

    def symmetric_orthogonalization(self, x):
        """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

        x: should have size [batch_size, 12]

        Output has size [batch_size, 3, 4], where each inner 3x4 matrix is [rot | trans].

        Taken directly from link provided by lecture slide.
        Original paper: Makadia, Ameesh. “An Analysis of SVD for Deep Rotation Estimation”. Advances in Neural Information Processing Systems 34. N.p., 2020. Print.
        Obtained code from: https://github.com/google-research/google-research/tree/master/special_orthogonalization
        """
        t_part = x[:, 9:].reshape(-1, 3, 1)
        rot = x[:, :9]
        m = rot.view(-1, 3, 3)
        u, s, v = torch.svd(m)
        vt = torch.transpose(v, 1, 2)
        det = torch.det(torch.matmul(u, vt))
        det = det.view(-1, 1, 1)
        vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
        r = torch.matmul(u, vt)
        ret = torch.cat([r, t_part], dim=-1)
        return ret


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
        x = nn.ReLU()(x)
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
