import os
import pandas as pd

def check_tags(s: str, tags: list) -> bool:
    return sum(list(map(lambda x: int(x in s), tags))) > 0

def get_config_mode(config_path, base_path=None):
    if base_path is not None:
        config_path = os.path.join(base_path, config_path)
    mode = config_path.split(".")[-2]
    if mode not in ["done", "in_progress"]: 
        return None
    return mode

def append_config_mode(config_path, mode, base_path=None):
    if base_path is not None:
        config_path = os.path.join(base_path, config_path)

    if get_config_mode(config_path) != mode:
        config_extension = config_path.split(".")[-1]
        config_path_without_extension = ".".join(config_path.split(".")[:-1]) if get_config_mode(config_path) is None else ".".join(config_path.split(".")[:-2])
        new_config_path = config_path_without_extension + "." + mode + "." + config_extension
        os.rename(
            config_path, 
            new_config_path
        )
        return new_config_path
    else:
        return config_path

def flatten_dict(d, root=""):
    d_flatten = {}

    for k, v in d.items():
        if isinstance(v, dict):
            d_flatten.update(flatten_dict(v, root=k))
        elif isinstance(v, list):
            for i, v_i in enumerate(v):
                d_flatten.update(flatten_dict({f"{i}": v_i}, root=k))
        else:
            key = root + "." + k if root != "" else k
            d_flatten[key] = v

    return d_flatten

def save_experiment_results(experiment_results : dict, results_path: str) -> None:
    experiment_results = pd.DataFrame({k: [v] for k, v in experiment_results.items()})
    if not os.path.exists(results_path): 
        results = experiment_results
    else: 
        results = pd.read_csv(results_path)
        results = pd.concat([results, experiment_results], axis=0)
    results.to_csv(results_path, index=False)