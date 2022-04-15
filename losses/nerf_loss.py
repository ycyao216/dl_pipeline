import torch
import numpy as np
import torch.nn as nn


class nerf_loss:
    def __init__(self):
        self.loss = self.corse_fine_loss

    def __call__(self, y_corse, y_fine, target):
        return self.loss(y_corse, y_fine, target)

    def corse_fine_loss(self, corse_r, fine_r, gt):
        c_loss = torch.mean(torch.cdist(corse_r, gt.squeeze(0)))
        # f_loss = torch.mean(torch.cdist(fine_r, gt.squeeze(0)))
        return c_loss.sum()


class nerf_metric:
    def __init__(self):
        self.batch_mIoU = 0

    def batch_accum(self, batch, output, label):
        return 0

    def epoch_result(self, dataset_size):
        return torch.tensor(0)
