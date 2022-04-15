from asyncio import run_coroutine_threadsafe
from distutils.command.config import config
import torch
import copy
from models import *
import torch.nn as nn
from utils import *
import tqdm
import pickle
import os
import optuna
import ffcv


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)


def prepare_model(device, program_config, configs=None, args=None) -> tuple:
    """Prepare the model and necessary optimization mechanisms

    Parameters
    ----------
    device : str
        Either cpu or Cuda
    program_config : dict
        Main configuration object for the entire porgram
    configs : dict, optional
        Model specific configuration specific for each run, by default None
    args : dict, optional
        Command line argument for chaning behavior of the program, by default None

    Returns
    -------
    tuple
        network, criterion, metric, optimizer, lr scheduler, run_function
    """
    # load model, criterion, optimizer, and learning rate scheduler
    model_name = configs["model_spec"]["model"]
    model_constructor = program_config["model_map"][model_name]
    network = model_constructor(configs).get_model()
    checkpoint_good_flag = True
    checkpoint_path = (
        os.path.join(args.checkpoint_dir, configs["model_spec"]["name"]) + ".pt"
    )
    checkpoint = None
    try:
        checkpoint = torch.load(checkpoint_path)
        if args.mode == 4:
            print("Hyper parameter tuning, removing finished checkpoint")
            os.remove(checkpoint_path)
            checkpoint = None
    except Exception as e:
        print("Cannot load checkpoint:" + checkpoint_path)
        checkpoint_good_flag = False

    try:
        for i in range(len(network)):
            network[i].load_state_dict(checkpoint["_model_state"][i])
        out_msg = (
            "=============================\n"
            + "=============================\n"
            + "Successfully loaded "
            + str(checkpoint_path)
            + ". Resuming."
        )
        print(out_msg)
    except Exception as e:
        if checkpoint_good_flag:
            out_msg = (
                "=============================\n"
                + "=============================\n"
                + "[WARNING] Load failed "
                + str(e)
                + ". Training will start from scrach."
            )
            for net in network:
                net.apply(initialize_weights)
            print(out_msg)
            checkpoint_good_flag = False

    task_name = configs["model_spec"]["task"]
    criterion_constructor = program_config["criterion_map"][task_name]
    metric_constructor = program_config["metric_map"][task_name]
    criterion = criterion_constructor()
    metric = metric_constructor()
    for i in range(len(network)):
        network[i] = network[i].to(device)
    # Set up optimizer
    optimizer = select_optimizer(configs, network)
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint["_optm_state"])
        except Exception as e:
            if checkpoint_good_flag:
                print("Failed to load optimizer state due to: " + str(e))
                response = input(
                    "Training may not proceed as expected, do you still want to continue? [y/N]"
                )
                if not (response.lower() == "y" or response.lower() == "yes"):
                    print("Terminated. Errors will be raised. Please check your files")
                    return None, None, None, None
    lr_scheduler = select_lr_sheculer(configs, optimizer)
    # Set up lr scheduler
    if checkpoint is not None:
        try:
            lr_scheduler.load_state_dict(checkpoint["_lr_state"])
        except Exception as e:
            if checkpoint_good_flag:
                print("Failed to load learning rate scheduler state due to: " + str(e))
                response = input(
                    "Training may not proceed as expected, do you still want to continue? [y/N]"
                )
                if not (response.lower() == "y" or response.lower() == "yes"):
                    print("Terminated. Errors will be raised. Please check your files")
                    return None, None, None, None
    try:
        run_function = program_config["run_modules"][model_name]
    except Exception as e:
        run_function = None
    if run_function is None:
        run_function = run_model
    return network, criterion, metric, optimizer, lr_scheduler, run_function


def run_model(
    data_loader, dataset_size, optimizer, model, criterion, metric, configs, is_train=0
):
    """Train or evaluate a model

    Parameters
    ----------
    data_loader : torch.Dataloader
        The dataloader to load the training or validation dataset
    optimizer : torch.optim
        The pytorch optimizer used by the model
    model : nn.Module
        The model to be trianed or tested
    criterion : nn.Loss
        The pytorch loss function to be used to get the performance of the model
    device : str
        The device on which the model will be run
    is_train : bool, optional
        If True, backprop will happen; if False, backprop will not happen, by default True

    Returns
    -------
    tuple
        (Cumulative loss over the entire dataset, cumulative correct predictions)
    """
    model = model[0]
    if is_train == 1 or is_train == 2:
        model.eval()
    cumulative_loss = 0
    for batch, labels in data_loader:
        if is_train == 0:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        if is_train == 3:
            return outputs, None
        loss = criterion(outputs, labels)
        cumulative_loss += loss
        metric.batch_accum(batch, outputs, labels)
        if is_train == 0:
            loss.backward()
            optimizer.step()
    return cumulative_loss.item() / dataset_size, metric.epoch_result(dataset_size)



def train_model(
    model,
    criterion,
    metric,
    optimizer,
    scheduler,
    device,
    dataloaders,
    run_model,
    configs=None,
    args=None,
    trial=None,
):
    """Train the entire model for all epochs

    Parameters
    ----------
    model : nn.Module
        Model to be trained
    criterion : function handel
        The loss function handle
    metric : function handle
        The metric function handle
    optimizer : nn.Optim
        Optimizer
    scheduler : nn.Optim.lr_scheduler
        learning rate scheduler
    device : str
        either cpu or cuda
    dataloaders : tuple
        (training dataloader, validating dataloader, testing dataloader)
    configs : dict, optional
        Model specific configuration, by default None
    args : dict, optional
        Run specific configuration, by default None
    """
    outputs = None
    epoch = configs["experiment"]["general"]["epoch"]
    # Initialize last loss to a large number
    last_loss = float(1e10)
    train_loader, val_loader, test_loader = dataloaders
    tadl, vdl, tsdl = 0, 0, 0
    if isinstance(train_loader, torch.utils.data.DataLoader):
        tadl = len(train_loader.dataset)
    elif isinstance(train_loader, ffcv.loader.loader.Loader):
        tadl = len(train_loader.indices)
    if isinstance(val_loader, torch.utils.data.DataLoader):
        vdl = len(val_loader.dataset)
    elif isinstance(val_loader, ffcv.loader.loader.Loader):
        vdl = len(val_loader.indices)
    if isinstance(test_loader, torch.utils.data.DataLoader):
        tsdl = len(test_loader.dataset)
    elif isinstance(test_loader, ffcv.loader.loader.Loader):
        tsdl = len(test_loader.indices)

    msg = (
        "Started training for "
        + configs["model_spec"]["model"]
        + " on device: "
        + str(device)
        + "\n"
        + "Optimizer: "
        + str(optimizer)
        + "\n"
        + "LR scheduler: "
        + str(scheduler)
        + "\n"
    )
    print(msg)
    # Assume save file have same name as model save file
    pickle_path = (
        os.path.join(args.result_dir, configs["model_spec"]["name"]) + "_results.pkl"
    )
    data = None
    checkpoint_path = (
        os.path.join(args.checkpoint_dir, configs["model_spec"]["name"]) + ".pt"
    )
    # Try to load results of previous attempt
    try:
        if args.mode != 4:
            with open(pickle_path, "rb") as inputfile:
                data = pickle.load(inputfile)
                msg = (
                    "Successfully loaded the following data: \n"
                    + str(data)
                    + " from: "
                    + pickle_path
                    + "\n"
                )
                print(msg)
                inputfile.close()
        else:
            data = {
                "training_loss": [],
                "training_acc": [],
                "validation_loss": [],
                "validation_acc": [],
                "epochs": 0,
                "testing_loss": None,
                "testing_acc": None 
            }
    except Exception as e:
        msg = (
            "[WARNING] Failed to load previous training results. New results pickle file will be made: "
            + pickle_path
            + ".\n"
        )
        print(msg)
        data = {
            "training_loss": [],
            "training_acc": [],
            "validation_loss": [],
            "validation_acc": [],
            "epochs": 0,
        }
    # train + val
    early_stop_flag = 0
    start = copy.deepcopy(data["epochs"])
    progress_bar = tqdm.tqdm(range(start, epoch))
    msg = ""
    for ep in progress_bar:
        if configs["experiment"]["general"]["early_stop"] and early_stop_flag > 3:
            break
        # Train the model
        train_loss, train_acc = run_model(
            train_loader, tadl, optimizer, model, criterion, metric, configs, is_train=0
        )
        msg = output_msg(
            train_loss, train_acc, data["epochs"], is_val=False, periodic=5
        )
        if args.mode != 4:
            print(msg)
        # Learning rate scheduler update
        scheduler.step()
        # Append trainig results
        data["training_loss"].append(train_loss)
        data["training_acc"].append(train_acc)

        with torch.no_grad():
            # Evaluate model
            val_loss, val_acc = run_model(
                val_loader,
                vdl,
                optimizer,
                model,
                criterion,
                metric,
                configs,
                is_train=1,
            )
            # Print results every 5 epochs.
            if args.mode != 4:
                msg = output_msg(
                    val_loss, val_acc, data["epochs"], is_val=True, periodic=5
                )
                print(msg)
            # Appending validation results
            data["validation_loss"].append(val_loss)
            data["validation_acc"].append(val_acc)
            if last_loss <= (val_loss + configs["experiment"]["general"]["es_penalty"]):
                early_stop_flag += 1
            else:
                early_stop_flag = 0
            last_loss = val_loss
            data["epochs"] = ep + 1
            if trial is not None:
                trial.report(val_loss, ep + 1)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        finished = ep + 1 >= epoch
        torch.save(
            {
                "_model_state": [m.state_dict() for m in model],
                "_optm_state": optimizer.state_dict(),
                "_lr_state": scheduler.state_dict(),
                "finished": finished,
            },
            checkpoint_path,
        )
        save_pickle(pickle_path, data)
    with torch.no_grad():
        # Test the model on test set
        has_gt = False 
        try: 
            has_gt = configs["meta"]["has_test_gt"]
        except Exception as e: 
            has_gt = False 
        if has_gt:
            test_loss, test_acc = run_model(
                test_loader, tsdl, optimizer, model, criterion, metric, configs, is_train=2
            )
            data["testing_loss"] = test_loss
            data["testing_acc"] = test_acc 
        else: 
            outputs, other_save = run_model(
                test_loader, tsdl, optimizer, model, criterion, metric, configs, is_train=3
            )
            if other_save:
                print("Saving is done by the run function. Please check the config file for save location and data format")
            if outputs is not None: 
                name = configs["model_spec"]["name"]
                save_dir = os.path.join(args.result_dir, name)
                save_dir += "_outputs.pkl"
                opt_data = {"model_name" : name, "output": outputs}
                save_pickle(save_dir, opt_data)
                print("Saved outputs to: " + save_dir)
        save_pickle(pickle_path, data)
        if args.mode != 4:
            msg = output_msg(test_loss, test_acc, data["epochs"],is_val = 2)
            print(msg)
    print("Training finished")
    return test_loss

def save_pickle(pickle_path, pkl_obj):
    with open(pickle_path, "wb") as output:
        pickle.dump(pkl_obj, output)
        output.close()

def output_msg(loss, accuracy, epoch, is_val=0, periodic=1):
    status = "Training"
    if is_val == 1:
        status = "Validating"
    elif is_val != 0:
        status = "Testing"
    out_msg_test = (
        "-----------------\n"
        "Finished "
        + status
        + ".\n"
        + status
        + " accuracy: "
        + str(accuracy * 100)
        + "%\n"
        + status
        + " loss: "
        + str(loss)
        + "\n"
        + "Printed after finishing epoch: "
        + str(epoch + 1)
        + ""
    )
    return out_msg_test
