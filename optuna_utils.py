
def optuna_config(trial, model_config):
    hyper_search_tree =model_config["hyper_search"]
    hyper_search_type = model_config["hyper_search_type"]
    hyper_search_range = model_config["hyper_search_range"]
    assert len(hyper_search_tree) == len(hyper_search_type)
    assert len(hyper_search_type) == len(hyper_search_range)
    for index, paths in enumerate(hyper_search_tree):
        root = model_config
        for level in range(len(paths)-1):
            root = root[paths[level]]
        root[paths[-1]] = optuna_config_substitute(paths[-1],trial,hyper_search_type[index],hyper_search_range[index])



def optuna_config_substitute(name, trial, dist_type:str, c_args):
    if dist_type == "loguniform":
        return trial.suggest_loguniform(name,*c_args)
    elif dist_type == "int":
        return trial.suggest_int(name,*c_args)
    elif dist_type == "categorical":
        return trial.suggest_categorical(name,c_args)
    else:
        return None