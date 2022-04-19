import json
import os
from engine import train_model, prepare_model
from data_utils import get_dataloader
import torch
import torchvision
import utils
import config_parser
import argparse
import optuna_utils
import optuna
import pandas 
import data_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
master_config_path = "./main_config.pkl"


def cross_validation(main_config, model_config, args, trial=None):
    m, criterion, metric, optimizer, lr_scheduler, run_func = prepare_model(
        device, main_config, model_config, args
    )
    train_data_loader, valid_data_loader, test_data_loader = get_dataloader(
        main_config, model_config
    )
    m = [mod.to(device) for mod in m]
    loss = train_model(
        m,
        criterion,
        metric,
        optimizer,
        lr_scheduler,
        device,
        (train_data_loader, valid_data_loader, test_data_loader),
        run_func,
        model_config,
        args,
        trial,
    )
    return loss


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
    m = [mod.to(device) for mod in m]
    visualizer = visualizer_const(m, test_data_loader)
    visualizer.visualize(model_config)


def apply_pre_processing(main_config, model_config, args):
    try:
        func = main_config["pre_preocessing"][model_config["model_spec"]["task"]]
    except Exception as e:
        func = None
    if func is not None:
        func(main_config, model_config, args)



def hyper_tuning(main_config, model_config, args):
    def objective(trial):
        optuna_utils.optuna_config(trial,model_config=model_config)
        loss = cross_validation(main_config=main_config, model_config=model_config, args=args, trial=trial)
        return loss 
    save_name = model_config["optuna"]["save_name"]
    storage = model_config["optuna"]["storage"]
    study = optuna.create_study(direction="minimize", 
                                sampler=optuna.samplers.TPESampler(), 
                                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                study_name=save_name,
                                storage=storage,
                                load_if_exists=True)
    study.optimize(objective, n_trials = model_config["optuna"]["trial_num"], gc_after_trial=True)
    return study 

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

def hyper_tuning(main_config, model_config, args):
    def objective(trial):
        optuna_utils.optuna_config(trial, model_config=model_config)
        loss = cross_validation(
            main_config=main_config, model_config=model_config, args=args, trial=trial
        )
        return loss

    print(model_config)
    save_name = model_config["optuna"]["save_name"]
    storage = model_config["optuna"]["storage"]
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        study_name=save_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        objective, n_trials=model_config["optuna"]["trial_num"], gc_after_trial=True
    )
    return study

def main(args):
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    model_config_dir = args.model_config_dir
    main_config = utils.load_pickle(master_config_path)
    data_root = main_config["database_root"]
    if not args.mode == 1:
        for model_config_pth in os.listdir(model_config_dir):
            try:
                model_file = os.path.join(model_config_dir, model_config_pth)
                with open(model_file, "r") as read_file:
                    model_config = json.load(read_file)
                    if model_config["meta"]["execute"] == False and args.mode != 5:
                        print(
                            "Skipping: "
                            + model_config["model_spec"]["name"]
                            + "\n=============================\n"
                        )
                        continue
                    prepend_root_path(data_root, model_config)
                    apply_pre_processing(main_config, model_config, args)
                    if args.mode == 3:
                        visualize_all(main_config, model_config, args)
                    elif args.mode == 4:
                        print(
                            "Hyperparameter tuning started. Model weight saves may not be the best model"
                        )
                        study = hyper_tuning(main_config, model_config, args)
                        print(study.best_params)
                        save_name = model_config["optuna"]["dataframe_name"]
                        df = study.trials_dataframe()
                        df.to_csv(save_name)
                    elif args.mode == 5:
                        study_name = model_config["optuna"]["save_name"]
                        study_storage = model_config["optuna"]["storage"]
                        try:
                            study_ = optuna.load_study(
                                study_name=study_name, storage=study_storage
                            )
                            print("Tuning summary for: " + study_name)
                            print(study_.best_trial)
                        except Exception as e:
                            print(
                                "Failed to read "
                                + study_name
                                + " stored at "
                                + study_storage
                                + ". Due to: "
                                + str(e)
                            )
                    else:
                        outputs = cross_validation(main_config, model_config, args)
            except Exception as e: 
                print("Failed to execute: " + model_config_pth + " due to: " +str(e))
                continue 
    if args.mode == 2 or args.mode == 1:
        utils.generate_all_plots(args.result_dir, args.save_dir)
        utils.generate_all_summary(args.result_dir, args.save_dir, args.summary_name)

config_parser.create_pickle_file()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default=0,
    type=int,
    help="Execution mode. 0 for train, 1 for plot, 2 for both train and plot, 3 for visualize outputs, 4 for hyperparameter tuning using optuna, 5 for reteriving resutls of optuna hyperparemeter tuning ",
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
    config_parser.create_pickle_file()
    main(args)
