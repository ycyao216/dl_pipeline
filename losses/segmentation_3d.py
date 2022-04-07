import torch
import numpy as np
import torch.nn as nn


class seg_3d_loss:
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, y, target):
        y = torch.transpose(y, -2, -1)
        return self.loss(y, target)


class seg_3d_metric:
    def __init__(self):
        self.batch_mIoU = 0

    def batch_accum(self, batch, output, label):
        return 0

    def epoch_result(self, dataset_size):
        return torch.tensor(0)
