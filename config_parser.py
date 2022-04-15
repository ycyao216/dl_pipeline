import pickle
from tkinter import NE

from datasets import pose_dataset, segment_3d_dataset, cifar_dataset, nerf_dataset
from losses import (
    categorical_classification,
    nerf_loss,
    pose_estimation_6d,
    segmentation_2d,
    segmentation_3d,
    pose_sym_aware_shape_agno_loss,
)
from models import (
    Nerf,
    PointNet_6dpose,
    PointNet_seg,
    Resnet,
    Resnet_cifar,
    Resnet_ref,
    Vision_transformer,
    Vit_ref,
)
from run_moduls import nerf_execute, pose_6d_execute
from visualizers import pose_6d_visualizer, segmentation_3d_visualizer

import torchvision
import optuna


master_config_path = "./main_config.pkl"

master_config_obj = {
    "database_root": "./",
    "model_map": {
        "resnet": Resnet.Resnet18,
        "vit": Vision_transformer.Vit_custom,
        "vit_ref": Vit_ref.ViT,
        "resnet_ref": Resnet_ref.ResNet,
        "resnet_cifar": Resnet_cifar.Resnet18_cifar,
        "PointNet_pose": PointNet_6dpose.PointNet,
        "PointNet_seg": PointNet_seg.PointNet,
        "Nerf": Nerf.Leveled_nerf,
    },
    "criterion_map": {
        "classification_cifar100": categorical_classification.cat_class_loss,
        "segmentation_3d": segmentation_3d.seg_3d_loss,
        "segmentation_2d": segmentation_2d.seg_2d_loss,
        "view_synthesys": nerf_loss.nerf_loss,
        "pose_estimation_6d": pose_sym_aware_shape_agno_loss.Error_calculator,
    },
    "metric_map": {
        "classification_cifar100": categorical_classification.cat_class_metric,
        "segmentation_3d": segmentation_3d.seg_3d_metric,
        "segmentation_2d": segmentation_2d.seg_2d_metric,
        "view_synthesys": nerf_loss.nerf_metric,
        "pose_estimation_6d": pose_sym_aware_shape_agno_loss.pose_estimator_metric,
    },
    "datasets": {
        "segmentation_3d": segment_3d_dataset.Segment_3d_dataset,
        "pose_estimation_6d": pose_dataset.PoseDataset,
        "classification_cifar100": cifar_dataset.cifar_wrapper,
        "view_synthesys": nerf_dataset.Nerf_dataset
    },
    "ffcv_writer": {
        "segmentation_3d": segment_3d_dataset.Segment_3d_ffcv_writer,
        "pose_estimation_6d": pose_dataset.Pose_dataset_ffcv_writer,
        "classification_cifar100": cifar_dataset.Cifar100_ffcv_writer,
    },
    "ffcv_loader": {
        "segmentation_3d": segment_3d_dataset.Segment_3d_ffcv_loader,
        "pose_estimation_6d": pose_dataset.Pose_dataset_ffcv_loader,
        "classification_cifar100": cifar_dataset.Cifar100_ffcv_loader,
    },
    "visualizer": {
        "None": None,
        "segmentation_3d": segmentation_3d_visualizer.segmentation_visualizer,
        "pose_estimation_6d": pose_6d_visualizer.Pose_box_visualizer,
        "classification_cifar100": None,
    },
    "run_modules": {
        "Nerf": nerf_execute.run_model,
        "pose_estimation_6d_with_norm": pose_6d_execute.run_model_with_normalization,
        "pose_estimation_6d_without_norm": pose_6d_execute.run_model_without_normalization,
        "Normal": None,
        "vit": None,
        "resnet": None
    },
    "pre_preocessing": {
        "pose_estimation_6d": None,
        "classification_cifar100":None,
        "view_synthesys": None
    },
    "pre_preocessing": {"pose_estimation_6d": None, "classification_cifar100": None},
}


def create_pickle_file():
    with open(master_config_path, "wb") as out_file:
        pickle.dump(master_config_obj, out_file)


if __name__ == "__main__":
    create_pickle_file()
