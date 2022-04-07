import json
import os
from engine import train_model, prepare_model
from data_utils import get_dataloader
import torch
import torchvision
import utils
import config_parser
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

master_config_path = "./main_config.pkl"


def cross_validation(main_config, model_config, args):
    if model_config["meta"]["execute"] == False:
        print(
            "Skipping: "
            + model_config["model_spec"]["name"]
            + "\n=============================\n"
        )
        return None
    m, criterion, metric, optimizer, lr_scheduler = prepare_model(
        device, main_config, model_config, args
    )
    train_data_loader, valid_data_loader, test_data_loader = get_dataloader(
        main_config, model_config
    )
    m = m.to(device)
    outputs = train_model(
        m,
        criterion,
        metric,
        optimizer,
        lr_scheduler,
        device,
        (train_data_loader, valid_data_loader, test_data_loader),
        model_config,
        args,
    )
    return outputs


def visualize_all(main_config, model_config, args):
    if model_config["meta"]["visualize"] == False:
        print(
            "Skipping: "
            + model_config["model_spec"]["name"]
            + "for visualization \n=============================\n"
        )
        return None
    m, __c, __m, __o, __l = prepare_model(device, main_config, model_config, args)
    __x, __y, test_data_loader = get_dataloader(main_config, model_config)
    visualizer_const = main_config["visualizer"][model_config["model_spec"]["task"]]
    m = m.to(device)
    visualizer = visualizer_const(m, test_data_loader)
    visualizer.visualize()


def apply_pre_processing(main_config, model_config, args):
    func = main_config["pre_preocessing"][model_config["model_spec"]["task"]]
    if func is not None:
        func(main_config, model_config, args)


def prepend_root_path(root_path, config_obj):
    config_obj["meta"]["training_dir_name"] = os.path.join(
        root_path, config_obj["meta"]["training_dir_name"]
    )
    config_obj["meta"]["testing_dir_name"] = os.path.join(
        root_path, config_obj["meta"]["testing_dir_name"]
    )
    config_obj["meta"]["ffcv_train_name"] = os.path.join(
        root_path, config_obj["meta"]["ffcv_train_name"]
    )
    config_obj["meta"]["ffcv_val_name"] = os.path.join(
        root_path, config_obj["meta"]["ffcv_val_name"]
    )
    config_obj["meta"]["ffcv_test_name"] = os.path.join(
        root_path, config_obj["meta"]["ffcv_test_name"]
    )


def main(args):
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    model_config_dir = args.model_config_dir
    main_config = utils.load_pickle(master_config_path)
    data_root = main_config["database_root"]
    if args.mode == 2 or args.mode == 0 or args.mode == 3:
        for model_config_pth in os.listdir(model_config_dir):
            model_file = os.path.join(model_config_dir, model_config_pth)
            with open(model_file, "r") as read_file:
                model_config = json.load(read_file)
                prepend_root_path(data_root, model_config)
                apply_pre_processing(main_config, model_config, args)
                if args.mode == 3:
                    visualize_all(main_config, model_config, args)
                else:
                    outputs = cross_validation(main_config, model_config, args)

    if args.mode == 2 or args.mode == 1:
        utils.generate_all_plots(args.result_dir, args.save_dir)
        utils.generate_all_summary(args.result_dir, args.save_dir, args.summary_name)


config_parser.create_pickle_file()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", default=0, type=int, help="Execution mode. 0 for train, 1 for plot"
)
parser.add_argument(
    "--save_dir",
    default="plot_results",
    type=str,
    help="The save directory of the generated plots",
)
parser.add_argument(
    "--model_dir", default="models", help="The directory of model constructors"
)
parser.add_argument(
    "--model_config_dir",
    default="model_configs",
    type=str,
    help="The directory of model configuration files",
)
parser.add_argument(
    "--summary_name",
    default="save_file.txt",
    type=str,
    help="Name of the summary txt file. Default to save_file.txt",
)
parser.add_argument(
    "--result_dir",
    default="./results/",
    type=str,
    help="The directory of the pkl files recording the resutls of runs. Default to resutls/",
)
parser.add_argument(
    "--checkpoint_dir",
    default="./checkpoints",
    type=str,
    help="The directory for saving the training checkpoints, default to checkpoints/",
)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
