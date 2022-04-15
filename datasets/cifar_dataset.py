from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    ToTorchImage,
    RandomTranslate,
    RandomHorizontalFlip,
    Convert,
    ModuleWrapper,
)
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import OrderOption
from ffcv.transforms.common import Squeeze
from torchvision import transforms
import torch
import torch.nn as nn
import os
import numpy as np
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ColorJitterWrapper(nn.Module):
    """Some Information about ColorJitterWrapper"""

    def __init__(self, configs):
        super(ColorJitterWrapper, self).__init__()
        self.configs = configs
        self.trans = transforms.ColorJitter(
            brightness=configs["augmentations"]["brightness"],
            hue=configs["augmentations"]["hue"],
            contrast=configs["augmentations"]["contrast"],
            saturation=configs["augmentations"]["saturation"],
        )

    def forward(self, x):
        x = self.trans(x)
        return x


class GrayScale(nn.Module):
    """Some Information about GrayScale"""

    def __init__(self):
        super(GrayScale, self).__init__()
        self.trans = transforms.Grayscale()

    def forward(self, x):
        x = self.trans(x)
        return x


class GaussianWrapper(nn.Module):
    """Some Information about GaussianWrapper"""

    def __init__(self, configs):
        super(GaussianWrapper, self).__init__()
        self.trans = transforms.GaussianBlur(
            kernel_size=configs["augmentations"]["kernel_size"],
            sigma=configs["augmentations"]["sigma"],
        )

    def forward(self, x):
        x = self.trans(x)
        return x


class Cifar_100_wraper(torch.utils.data.Dataset):
    """Some Information about Cifar_100_wraper"""

    def __init__(self):
        super(Cifar_100_wraper, self).__init__()

    def __getitem__(self, index):
        return

    def __len__(self):
        return


def create_dataloader(path, configs):
    decoder = SimpleRGBImageDecoder()
    augment_pipeline = [
        decoder,
        RandomTranslate(padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        ToTorchImage(),
    ]
    if (
        configs["augmentations"]["brightness"] != 0.0
        and configs["augmentations"]["hue"] != 0.0
        and configs["augmentations"]["contrast"] != 0.0
        and configs["augmentations"]["saturation"] != 0.0
    ):
        augment_pipeline.append(ModuleWrapper(ColorJitterWrapper(configs)))
    if configs["augmentations"]["grayscale"]:
        augment_pipeline.append(ModuleWrapper(transforms.Grayscale(3)))
    if len(configs["augmentations"]["kernel_size"]) > 0 and len(configs["sigma"]) > 0:
        augment_pipeline.append(ModuleWrapper(GaussianWrapper(configs)))

    to_img_pipeline = [Convert(torch.float32), ToDevice(device, non_blocking=True)]
    image_pipeline = augment_pipeline
    image_pipeline.extend(to_img_pipeline)
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    pipelines = {"image": image_pipeline, "label": label_pipeline}
    loader = Loader(
        path,
        batch_size=configs["experiment"]["general"]["bz"],
        num_workers=8,
        order=OrderOption.RANDOM,
        pipelines=pipelines,
    )
    return loader


def create_classification_beton_file(save_path, resolution=32):
    writer = DatasetWriter(
        save_path,
        {
            # Tune options to optimize dataset size, throughput at train-time
            "image": RGBImageField(max_resolution=resolution),
            "label": IntField(),
        },
    )
    return writer


def cifar_wrapper(config, train=0):
    if train == 0 or train == 1:
        return torchvision.datasets.CIFAR100(
            root=config["meta"]["dataset_path"], train=True, download=True
        )
    elif train == 2:
        return torchvision.datasets.CIFAR100(
            root=config["meta"]["dataset_path"], train=False, download=True
        )


class Cifar100_ffcv_writer:
    def __init__(self, config):
        self.config = config
        pass

    def make_writer(self, path):
        return create_classification_beton_file(path)


class Cifar100_ffcv_loader:
    def __init__(self, config):
        self.config = config

    def make_loader(self, beton_path, config):
        return create_dataloader(beton_path, config)
