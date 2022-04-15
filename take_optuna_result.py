import optuna
import torch
import os
import pickle
import json

for i in os.listdir("./vit_tuning"):
    dirt = os.path.join("./vit_tuning", i)
    with open(dirt, "r") as in_json:
        config = json.load(in_json)
        name = config["optuna"]["save_name"]
        storage = config["optuna"]["storage"]
        study = optuna.load_study(study_name=name, storage=storage)
        print(study.best_trial)
