from cProfile import label
from torch.utils.data import dataset, dataloader
import os
import numpy as np
import torch
from . import DEVICE
import tqdm
from random import sample, shuffle


class Segment_3d_dataset(torch.utils.data.Dataset):
    """Some Information about Segment_3d_dataset"""

    def __init__(self, config_file, train=True):
        super(Segment_3d_dataset, self).__init__()
        data_base_dir = os.path.join(
            config_file["meta"]["dataset_path"],
            config_file["meta"]["training_dir_name"],
        )
        if not train:
            data_base_dir = os.path.join(
                config_file["meta"]["dataset_path"],
                config_file["meta"]["testing_dir_name"],
            )

        pts_dir = os.path.join(data_base_dir, "pts")
        label_dir = os.path.join(data_base_dir, "label")
        self.labels = []
        self.pts = []
        for i in os.listdir(pts_dir):
            base_name = i.split(".")[0]
            pts_file = os.path.join(pts_dir, base_name + ".pts")
            self.pts.append(self.read_pts(pts_file))
            if train:
                label_name = os.path.join(label_dir, base_name + ".txt")
                self.labels.append(self.read_label(label_name))
            else:
                self.labels.append(torch.zeros(2048))

        for index in tqdm.tqdm(range(len(self.pts))):
            if len(self.pts[index]) < 2048:
                self.pts[index].extend(
                    [self.pts[index][0]] * (2048 - len(self.pts[index]))
                )
                self.pts[index] = torch.tensor(self.pts[index])
                if train:
                    self.labels[index].extend(
                        [self.labels[index][0]] * (2048 - len(self.labels[index]))
                    )
                    self.labels[index] = torch.tensor(
                        self.labels[index], dtype=torch.long
                    )
            else:
                self.pts[index] = torch.tensor(self.pts[index])
                self.labels[index] = torch.tensor(self.labels[index], dtype=torch.long)
                indices = list(range(len(self.pts[index])))
                shuffle(indices)
                indices = torch.tensor(indices[:2048])
                self.pts[index] = self.pts[index][indices]
                if train:
                    self.labels[index] = self.labels[index][indices]

    def read_label(self, label_path):
        labels = []
        with open(label_path, "r") as in_file:
            lines = in_file.readlines()
            labels = [int(i) - 1 for i in lines]
            in_file.close()
        return labels

    def read_pts(self, file_path):
        point_cloud = []
        with open(file_path, "r") as in_file:
            lines = in_file.readlines()
            for i in lines:
                seperated = [float(j) for j in i.split()]
                point_cloud.append(seperated)
            in_file.close()
        return point_cloud

    def __getitem__(self, index):
        return self.pts[index].to(DEVICE), self.labels[index].to(DEVICE)

    def __len__(self):
        return len(self.labels)


class Segment_3d_ffcv_writer:
    def __init__(self, config):
        pass

    def __call__(self):
        pass


class Segment_3d_ffcv_loader:
    def __init__(self, config):
        pass

    def __call__(self, beton_path, config):
        pass
