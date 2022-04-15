from torch.utils.data import dataloader, dataset
import os
import tqdm
import json
import numpy as np
import pickle
import PIL.Image as Image
from ffcv.fields import RGBImageField, NDArrayField, IntField
from ffcv.fields.decoders import SimpleRGBImageDecoder, NDArrayDecoder, IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage
from ffcv.writer import DatasetWriter
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PoseDataset(dataset.Dataset):
    """6d pose estimation dataset"""

    def __init__(self, config, train=0):
        super(PoseDataset, self).__init__()
        self.train_objs = []
        if train == 0:
            pickle_file_path = config["training_dir_name"]
        elif train == 1:
            pickle_file_path = config["validating_dir_name"]
        else:
            pickle_file_path = config["testing_dir_name"]

        files = os.listdir(pickle_file_path)
        if subset_size == -1:
            subset_size = len(files)
        for i in tqdm.tqdm(files[:subset_size]):
            open_dir = os.path.join(pickle_file_path, i)
            self.train_objs.append(load_pickle(open_dir))
        self.point_size = config["special"]["point_size"]

    def __getitem__(self, index):
        obj_id = self.train_objs[index]["object_id"]
        scan_points = get_point_cloud_from_scan_file(
            self.train_objs[index]["depth_file"],
            self.train_objs[index]["label_file"],
            obj_id,
            self.train_objs[index]["intrinsic"],
            self.train_objs[index]["extrinsic"],
            self.point_size,
        )
        gt_transform = self.train_objs[index]["pose_world"]
        return scan_points, gt_transform, obj_id

    def __len__(self):
        return len(self.train_objs)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def depth_to_points_np(depth, intrinsic, extrinsic):
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    points_viewer = points_viewer.reshape((-1, 3))
    points_viewer = points_viewer[~np.all(points_viewer == 0, axis=1)]

    points_viewer = homogenous_tranform(points_viewer, np.linalg.inv(extrinsic))
    return points_viewer


def fps_downsample_np(num_points: int, input_cloud: np.ndarray):
    """Get the mesh from a dae file and texture file, and then down sample if
    necessary. The downsample happenes when the number of point is smaller to
    the number of vertices in the mesh

    Parameters
    ----------
    num_points : int
        The number of points to downsample
    mesh_path : str
        The file location of the dae file
    meta : dict
        The meta dictionary
    Returns
    -------
    np.ndarray
        A numpy array containing the point cloud vertices of the mesh
    """
    selected_points = np.zeros((num_points, 3))
    dist = np.ones(input_cloud.shape[0]) * np.inf
    for i in range(num_points):
        idx = np.argmax(dist)
        selected_points[i] = input_cloud[idx]
        dist_ = ((input_cloud - selected_points[i]) ** 2).sum(-1)
        dist = np.minimum(dist, dist_)
    return selected_points


def random_padding(num_points: int, input_cloud: np.ndarray):
    if input_cloud.shape[0] == 0:
        print(input_cloud)
    padded_points = np.zeros((num_points, input_cloud.shape[1]))
    padded_points[: input_cloud.shape[0], :] = input_cloud
    missing = num_points - input_cloud.shape[0]
    idx = np.random.randint(input_cloud.shape[0], size=missing)
    padded_points[input_cloud.shape[0] :, :] = input_cloud[idx]
    return padded_points


def fast_down_sample(point_size, scan_point_cloud):
    return scan_point_cloud[:point_size, :]


def random_padding_fast(num_points, input_cloud):
    new_arr = np.repeat(input_cloud[:1, :], num_points, axis=0)
    new_arr[: input_cloud.shape[0], :] = input_cloud
    return new_arr


def get_point_cloud_from_scan_file(
    scan_file_path: str,
    label_path: str,
    object_id: int,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    point_size,
):
    """Generate a scaned point cloud as np array. The scanned point cloud is only
    for a single item in a scene

    Parameters
    ----------
    scan_file_path : str
        The file containing scanned point depth map image
    label_path : str
        The file containing labeled segmentation map for the scene
    meta_path : str
        The meta file path of the scene
    object_id : int
        The object id of the object whose pose and scan will be returned
    """
    scan_point_cloud = np.array(Image.open(scan_file_path)) / 1000
    if isinstance(label_path, str):
        label_map = np.array(Image.open(label_path))
    else:
        label_map = label_path
    scan_point_cloud = np.where(label_map == object_id, scan_point_cloud, 0)
    if np.all(scan_point_cloud == 0):
        return None
    scan_point_cloud = depth_to_points_np(scan_point_cloud, intrinsics, extrinsics)
    if scan_point_cloud.shape[0] == 0:
        return scan_point_cloud
    if scan_point_cloud.shape[0] < point_size:
        scan_point_cloud = random_padding(point_size, scan_point_cloud)
    elif scan_point_cloud.shape[0] > point_size:
        scan_point_cloud = fps_downsample_np(point_size, scan_point_cloud)
    return scan_point_cloud.astype(np.float32)


def to_homogenous(input_mat: np.ndarray) -> np.ndarray:
    """Move a input Nx3 matrix to Nx4 homogenous coordinates

    Parameters
    ----------
    input_mat : np.ndarray
        Nx3 input point cloud

    Returns
    -------
    np.ndarray
        Output point cloud in homogenous coordinates Nx4
    """
    homogenous = np.concatenate((input_mat, np.ones((input_mat.shape[0], 1))), axis=1)
    return homogenous


def de_homogenous(input_mat: np.ndarray) -> np.ndarray:
    """Move a Nx4 homogenous coordinate point cloud to Nx3 coordinates

    Parameters
    ----------
    input_mat : np.ndarray
        Input point cloud Nx4

    Returns
    -------
    np.ndarray
        Nx3 point cloud
    """
    return input_mat[:, :3] / input_mat[:, -1:]


def homogenous_tranform(
    input_mat: np.ndarray, transformation: np.ndarray
) -> np.ndarray:
    """Conduct transformation in homogenous space, will be raised to homogenous space

    Parameters
    ----------
    input_mat : np.ndarray
        Nx3 input array.
    transformation : np.ndarray
        4x4 transformation matrix

    Returns
    -------
    np.ndarray
        Nx3 transformed array
    """
    input_mat = to_homogenous(input_mat)
    input_mat = (transformation @ input_mat.T).T
    input_mat = de_homogenous(input_mat)
    return input_mat


class Pose_dataset_ffcv_writer:
    def __init__(self, config):
        self.pointsize = config["special"]["point_size"]
        self.num_worker = 4

    def make_writer(self, save_path):
        writer = DatasetWriter(
            save_path,
            {
                "pointcloud": NDArrayField(
                    shape=(self.pointsize, 3), dtype=np.dtype("float32")
                ),
                "worldpose": NDArrayField(shape=(4, 4), dtype=np.dtype("float32")),
                "obj_id": IntField(),
            },
            num_workers=self.num_workers,
        )
        return writer


class Pose_dataset_ffcv_loader:
    def __init__(self, config):
        self.subset_length = config["data"]["subset_fraction"]
        self.batch_size = config["experiment"]["general"]["bz"]
        self.num_workers = 4

    def make_loader(self, beton_path):
        ORDERING = OrderOption.RANDOM
        PIPELINES = {
            "pointcloud": [
                NDArrayDecoder(),
                ToTensor(),
                ToDevice(device, non_blocking=True),
            ],
            "worldpose": [
                NDArrayDecoder(),
                ToTensor(),
                ToDevice(device, non_blocking=True),
            ],
            "obj_id": [IntDecoder()],
        }
        loader = Loader(
            beton_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pipelines=PIPELINES,
            order=ORDERING,
        )
        return loader
