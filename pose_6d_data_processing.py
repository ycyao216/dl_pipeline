import numpy as np
import tqdm
import os
import pickle
import numpy as np
import PIL.Image as Image


def load_pickle(pkl_path):
    with open(pkl_path, "rb") as in_file:
        obj = pickle.load(in_file)
        return obj


def generate_compact_reps(
    rgb_files, depth_files, label_files, meta_files, train, save_path
):
    compact_obj = {}
    total = len(rgb_files)
    with tqdm.tqdm(total=total) as progres_bar:
        for rgb, depth, label, meta in zip(
            rgb_files, depth_files, label_files, meta_files
        ):
            if ~(
                os.path.basename(os.path.normpath(meta)).split("_")[0][0]
                in ["1", "2", "3"]
            ):
                progres_bar.update(1)
                meta_obj = load_pickle(meta)
                for objs in meta_obj["object_ids"].tolist():
                    save_name = os.path.join(
                        save_path,
                        os.path.basename(os.path.normpath(meta)).split("_")[0]
                        + "_compact"
                        + str(objs)
                        + ".pkl",
                    )
                    if not os.path.exists(save_name) and np.any(
                        np.unique(np.array(Image.open(label))) == objs
                    ):
                        compact_obj["rgb_file"] = rgb
                        compact_obj["depth_file"] = depth
                        compact_obj["label_file"] = label
                        compact_obj["extrinsic"] = meta_obj["extrinsic"]
                        compact_obj["intrinsic"] = meta_obj["intrinsic"]
                        compact_obj["object_id"] = objs
                        if train:
                            compact_obj["pose_world"] = meta_obj["poses_world"][objs]
                        else:
                            compact_obj["pose_world"] = None
                        compact_obj["scale"] = meta_obj["scales"][objs]
                        compact_obj["name"] = os.path.basename(
                            os.path.normpath(meta)
                        ).split("_")[0]
                        with open(save_name, "wb+") as handle:
                            pickle.dump(
                                compact_obj, handle, protocol=pickle.HIGHEST_PROTOCOL
                            )
                            handle.close()


def generate_compact_reps_wrapper(
    data_dir, split_name, split_dir, save_path, train=True
):
    rgb, depth, label, meta = get_data_paths(data_dir, split_name, split_dir, train)
    generate_compact_reps(rgb, depth, label, meta, train, save_path)


def pre_process():
    main_config = load_pickle("./main_config.pkl")
    data_root = os.path.join(main_config["database_root"], "pose_estimation_dataset")
    training_data_dir = os.path.join(data_root, "training_data/v2.2/")
    train_split_name = "train"
    val_split_name = "val"
    train_split_dir = os.path.join(data_root, "training_data/splits/v2/")
    testing_data_dir = os.path.join(data_root, "testing_data/v2.2/")
    obj_file_path = os.path.join(data_root, "training_data/objects_v1.csv")
    save_train = os.path.join(data_root, "pose_training/")
    save_val = os.path.join(data_root, "pose_validating/")
    save_test = os.path.join(data_root, "pose_testing/")
    if not os.path.exists(save_train):
        os.mkdir(save_train)
    if not os.path.exists(save_val):
        os.mkdir(save_val)
    if not os.path.exists(save_test):
        os.mkdir(save_test)
    generate_compact_reps_wrapper(
        training_data_dir, train_split_name, train_split_dir, save_train, train=True
    )
    generate_compact_reps_wrapper(
        training_data_dir, val_split_name, train_split_dir, save_val, train=True
    )
    generate_compact_reps_wrapper(
        testing_data_dir, train_split_name, train_split_dir, save_test, train=False
    )


def directly_get_files(testing_dir):
    dataid = set()
    for i in os.listdir(testing_dir):
        # @NOTE: Only taking level 1-3 for now.
        if (
            i.split("_")[0].split("-")[0] == "1"
            or i.split("_")[0].split("-")[0] == "2"
            or i.split("_")[0].split("-")[0] == "3"
        ):
            dataid.add(i.split("_")[0])
    rgb = [os.path.join(testing_dir, p + "_color_kinect.png") for p in dataid]
    depth = [os.path.join(testing_dir, p + "_depth_kinect.png") for p in dataid]
    label = [os.path.join(testing_dir, p + "_label_kinect.png") for p in dataid]
    meta = [os.path.join(testing_dir, p + "_meta.pkl") for p in dataid]
    return rgb, depth, label, meta


def get_split_files(training_data_dir, split_name, split_dir):
    with open(os.path.join(split_dir, f"{split_name}.txt"), "r") as f:
        prefix = [
            os.path.join(training_data_dir, line.strip()) for line in f if line.strip()
        ]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta


def get_data_paths(data_dir, split_name, split_dir, train=True):
    if train:
        return get_split_files(data_dir, split_name, split_dir)
    return directly_get_files(data_dir)


if __name__ == "__main__":
    pre_process()
