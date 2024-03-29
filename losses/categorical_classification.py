import torch
import numpy as np
import torch.nn as nn


class cat_class_loss:
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, y, target):
        return torch.nan_to_num(self.loss(y, target), nan=1e10, posinf=1e10,neginf=1e10)


class cat_class_metric:
    def __init__(self):
        self.correct_preds = 0

    def batch_accum(self, batch, output, label):
        _, predicted = torch.max(output.data, 1)
        self.correct_preds += (predicted == label).sum()

    def epoch_result(self, dataset_size):
        retval = self.correct_preds.item() / dataset_size
        self.correct_preds = 0
        return retval
