# from ffcv.writer import DatasetWriter
# from ffcv.fields import RGBImageField, IntField
# from ffcv.loader import Loader, OrderOption
# from ffcv.transforms import (
#     ToTensor,
#     ToDevice,
#     ToTorchImage,
#     RandomTranslate,
#     RandomHorizontalFlip,
#     Convert,
#     ModuleWrapper,
# )
# from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
# from ffcv.loader import OrderOption
# from ffcv.transforms.common import Squeeze
from csv import writer
from torchvision import transforms
import torch
import torch.nn as nn
import os
import numpy as np
import pathlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ColorJitterWrapper(nn.Module):
    """Some Information about ColorJitterWrapper"""

    def __init__(self, args):
        super(ColorJitterWrapper, self).__init__()
        self.trans = transforms.ColorJitter(
            brightness=args["augmentations"]["brightness"],
            hue=args["augmentations"]["hue"],
            contrast=args["augmentations"]["contrast"],
            saturation=args["augmentations"]["saturation"],
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

    def __init__(self, args):
        super(GaussianWrapper, self).__init__()
        self.trans = transforms.GaussianBlur(
            kernel_size=args["augmentations"]["kernel_size"],
            sigma=args["augmentations"]["sigma"],
        )

    def forward(self, x):
        x = self.trans(x)
        return x

def prepend_root_path(root_path, config_obj):
    config_obj["meta"]["training_dir_name"] = str(pathlib.PurePath(os.path.join(
        root_path, config_obj["meta"]["training_dir_name"]
    )))
    config_obj["meta"]["validating_dir_name"] = str(pathlib.PurePath(os.path.join(
        root_path, config_obj["meta"]["validating_dir_name"]
    )))
    config_obj["meta"]["testing_dir_name"] = str(pathlib.PurePath(os.path.join(
        root_path, config_obj["meta"]["testing_dir_name"]
    )))
    config_obj["meta"]["ffcv_train_name"] = str(pathlib.PurePath(os.path.join(
        root_path, config_obj["meta"]["ffcv_train_name"]
    )))
    config_obj["meta"]["ffcv_val_name"] = str(pathlib.PurePath(os.path.join(
        root_path, config_obj["meta"]["ffcv_val_name"]
    )))
    config_obj["meta"]["ffcv_test_name"] = str(pathlib.PurePath(os.path.join(
        root_path, config_obj["meta"]["ffcv_test_name"]
    )))

def get_dataloader(master_config, config):
    data_base_root = pathlib.Path(
        os.path.join(master_config["database_root"], config["meta"]["dataset_path"])
    ).resolve()
    prepend_root_path(data_base_root, config)
    dataset_constr = master_config["datasets"][config["model_spec"]["task"]]
    training_dataset, testing_dataset, validating_dataset = None, None, None
    ffcv_train_path = config["meta"]["ffcv_train_name"]
    ffcv_val_path = config["meta"]["ffcv_val_name"]
    ffcv_test_path = config["meta"]["ffcv_test_name"]
    if config["data"]["train_fraction"] == 1.0:
        ffcv_val_path = ffcv_test_path

    if config["meta"]["use_ffcv"] == False:
        training_dataset = dataset_constr(config, train=0)
        testing_dataset = dataset_constr(config, train=2)
        validating_dataset = testing_dataset
        if config["data"]["train_fraction"] != 1.0:
            train_set_len = int(
                len(training_dataset) * config["data"]["train_fraction"]
            )
            val_set_len = len(training_dataset) - train_set_len
            training_dataset, validating_dataset = torch.utils.data.random_split(
                training_dataset, [train_set_len, val_set_len]
            )
        elif (
            config["meta"]["validating_dir_name"] is not None
            or config["meta"]["validating_dir_name"] != ""
        ):
            validating_dataset = dataset_constr(config, train=1)

        if config["data"]["subset_fraction"] != 1.0:
            indices = np.arange(len(training_dataset))
            np.random.shuffle(indices)
            indices = indices[
                : int(len(training_dataset) * config["data"]["subset_fraction"])
            ]
            training_dataset = torch.utils.data.Subset(training_dataset, indices)
            print(
                "Training dataset will contain {} samples.".format(
                    str(len(training_dataset))
                )
            )
    else:
        ffcv_datawriter = master_config["ffcv_writer"][config["model_spec"]["task"]](
            config
        )
        ffcv_dataloader = master_config["ffcv_loader"][config["model_spec"]["task"]](
            config
        )
        if not os.path.exists(ffcv_train_path):
            label_writer = ffcv_datawriter.make_writer(ffcv_train_path)
            training_dataset = dataset_constr(config, train=0)
            if config["data"]["subset_fraction"] != 1.0:
                indices = np.arange(len(training_dataset))
                np.random.shuffle(indices)
                indices = indices[
                    : int(len(training_dataset) * config["data"]["subset_fraction"])
                ]
                training_dataset_sub = torch.utils.data.Subset(training_dataset, indices)
                print(
                    "Training dataset will contain {} samples.".format(
                        str(len(training_dataset_sub))
                    )
                )
                label_writer.from_indexed_dataset(training_dataset_sub)
            else: 
                label_writer.from_indexed_dataset(training_dataset)

        if not os.path.exists(ffcv_test_path):
            non_label_writer = ffcv_datawriter.make_writer(ffcv_test_path)
            testing_dataset = dataset_constr(config, train=2)
            non_label_writer.from_indexed_dataset(testing_dataset)

        if not os.path.exists(ffcv_val_path):
            if config["data"]["train_fraction"] != 1.0:
                train_set_len = int(
                    len(training_dataset) * config["data"]["train_fraction"]
                )
                val_set_len = len(training_dataset) - train_set_len
                training_dataset, validating_dataset = torch.utils.data.random_split(
                    training_dataset, [train_set_len, val_set_len]
                )
            elif (
                config["meta"]["validating_dir_name"] is not None
                or config["meta"]["validating_dir_name"] != ""
            ):
                validating_dataset = dataset_constr(config, train=1)
            else: 
                validating_dataset = testing_dataset
            label_writer = ffcv_datawriter.make_writer(ffcv_val_path)
            label_writer.from_indexed_dataset(validating_dataset)

        train_loader = ffcv_dataloader.make_loader(ffcv_train_path, config)
        val_loader = ffcv_dataloader.make_loader(ffcv_val_path, config)
        test_loader = ffcv_dataloader.make_loader(ffcv_test_path, config)
        return train_loader, val_loader, test_loader

    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=config["experiment"]["general"]["bz"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        validating_dataset,
        batch_size=config["experiment"]["general"]["bz"],
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testing_dataset, batch_size=config["experiment"]["general"]["bz"], shuffle=True
    )
    return train_loader, val_loader, test_loader
