import torch
import os
import pickle
import matplotlib.pyplot as plt


def load_pickle(pkl_path):
    with open(pkl_path, "rb") as in_file:
        obj = pickle.load(in_file)
        return obj


def select_lr_sheculer(configs, optimizer):
    scheduler = None
    if "multistep" in configs["experiment"]["scheduler"]["lr_scheduler"].lower():
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=configs["experiment"]["scheduler"]["mile_stones"],
            gamma=configs["experiment"]["scheduler"]["gamma"],
            verbose=False,
        )
    elif "linear" in configs["experiment"]["scheduler"]["lr_scheduler"].lower():
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=1e-9,
            total_iters=configs["experiment"]["scheduler"]["epoch"],
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    return scheduler


def select_optimizer(configs, network):
    weights = [n.parameters() for n in network]
    concatenated = []
    for i in weights:
        concatenated += list(i)
    if "sgd" in configs["experiment"]["optimizer"]["optimizer"].lower():
        optimizer = torch.optim.SGD(
            concatenated,
            lr=configs["experiment"]["general"]["lr"],
            weight_decay=configs["experiment"]["optimizer"]["weight_decay"],
            momentum=configs["experiment"]["optimizer"]["momentum"],
            dampening=configs["experiment"]["optimizer"]["dampening"],
        )
    elif "adam" in configs["experiment"]["optimizer"]["optimizer"].lower():
        optimizer = torch.optim.Adam(
            concatenated,
            lr=configs["experiment"]["general"]["lr"],
            weight_decay=configs["experiment"]["optimizer"]["weight_decay"],
        )
    else:
        optimizer = None
    return optimizer


def get_summary(pkl_file: str) -> tuple:
    """Get summary of the experiment resutls

    Parameters
    ----------
    pkl_file : str
        Pickle file directory

    Returns
    -------
    tuple
        training loss, training accuracy, validation loss, validation accuracy
    """
    data = load_pickle(pkl_file)
    last_t_loss = (
        data["training_loss"][-1]
        if len(data["training_loss"]) <= 150
        else data["training_loss"][149]
    )
    last_t_acc = (
        data["training_acc"][-1]
        if len(data["training_acc"]) <= 150
        else data["training_acc"][149]
    )
    last_v_loss = (
        data["validation_loss"][-1]
        if len(data["validation_loss"]) <= 150
        else data["validation_loss"][149]
    )
    last_v_acc = (
        data["validation_acc"][-1]
        if len(data["validation_acc"]) <= 150
        else data["validation_acc"][149]
    )
    return last_t_loss, last_t_acc, last_v_loss, last_v_acc


def generate_all_summary(pkl_file_dir, save_directory, save_name):
    min_tl = 1e10
    max_ta = 0
    min_vl = 1e10
    max_va = 0
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    output_name = os.path.join(save_directory, save_name)
    output_file = open(output_name, "w")
    for i in os.listdir(pkl_file_dir):
        name = os.path.join(pkl_file_dir, i)
        if ".pkl" in i:
            tl, ta, vl, va = get_summary(name)
            tl = tl.item() if not isinstance(tl, float) else tl
            ta = ta.item() if not isinstance(ta, float) else ta
            vl = vl.item() if not isinstance(vl, float) else vl
            va = va.item() if not isinstance(va, float) else va
            msg = "Result of {}:\n Training loss {};\n Training accuracy{};\n Validating loss {};\n Validating accuracy{}\n================================\n".format(
                name, str(tl), str(ta), str(vl), str(va)
            )
            print(msg)
            output_file.write(msg)
            min_tl = tl if tl < min_tl else min_tl
            max_ta = ta if ta > max_ta else max_ta
            min_vl = vl if vl < min_vl else min_vl
            max_va = va if va > max_va else max_va


def generate_plots(plot_name, pkl_file, save_directory):
    data = load_pickle(pkl_file)
    for key in data:
        try:
            for idx, item in enumerate(data[key]):
                data[key][idx] = item.item()
        except Exception as exp:
            break

    training_loss = (
        data["training_loss"]
        if len(data["training_loss"]) <= 150
        else data["training_loss"][:150]
    )
    training_acc = (
        data["training_acc"]
        if len(data["training_acc"]) <= 150
        else data["training_acc"][:150]
    )
    validation_loss = (
        data["validation_loss"]
        if len(data["validation_loss"]) <= 150
        else data["validation_loss"][:150]
    )
    validation_acc = (
        data["validation_acc"]
        if len(data["validation_acc"]) <= 150
        else data["validation_acc"][:150]
    )
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    (train_loss,) = ax[0].plot(training_loss, label="Training Loss")
    (valid_loss,) = ax[0].plot(validation_loss, label="Test Loss")
    ax[0].set_xlabel("Epochs", fontsize=40)
    ax[0].set_ylabel("Loss", fontsize=40)
    ax[0].legend()
    (train_acc,) = ax[1].plot(training_acc, label="Training Accuracy")
    (valid_acc,) = ax[1].plot(validation_acc, label="Test Accuracy")
    ax[1].set_xlabel("Epochs", fontsize=40)
    ax[1].set_ylabel("Accuracy", fontsize=40)
    ax[1].legend()
    plt.suptitle("Performance curve of: " + plot_name, fontsize=40)
    plt.savefig(save_directory)
    plt.clf()


def generate_all_plots(pkl_file_dir, save_directory):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    for pkl_file in os.listdir(pkl_file_dir):
        if ".pkl" in pkl_file:
            print("Generating plot for: " + pkl_file)
            pkl_file = os.path.join(pkl_file_dir, pkl_file)
            name = os.path.basename(os.path.normpath(pkl_file)).split(".")[0]
            name = name.replace("own", "")
            name = name.replace("ref", "")
            save_name = os.path.join(save_directory, name) + "_plots.JPG"
            try:
                generate_plots(name, pkl_file, save_name)
            except Exception as e:
                print(
                    "Failed due to: " + str(e) + "\n================================="
                )
                continue
            print("Success.\n======================================")


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
