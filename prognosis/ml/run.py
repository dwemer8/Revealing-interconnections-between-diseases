import os
import datetime
import json

import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

from src.automl_trainer import AutomlTrainer
from src.dummy_trainer import DummyTrainer
from src.sklearn_trainer import SklearnTrainer
from src.utils import flatten_dict, save_experiment_results

@hydra.main(version_base=None, config_path="configs", config_name="run")
def run(cfg : DictConfig) -> None:
    if not OmegaConf.has_resolver("eval"): OmegaConf.register_new_resolver("eval", eval) #arithmetic in config params

    print("------------------------------------------------")
    print("Experiment", cfg.base.experiment_name, "started")
    print("------------------------------------------------")
    print(json.dumps(
        OmegaConf.to_container(cfg, resolve=True),
        indent=4,
        ensure_ascii=False
    ))
    datetime_start = datetime.datetime.now()
    
    #data reading
    print("Reading data...")
    dataset = f"{cfg.data.nosology}{cfg.data.dataset_suffix}"
    file_path = cfg.data.work_dir + dataset
    if ".xlsx" in file_path:
        df = pd.read_excel(file_path)
    elif ".csv" in file_path:
        df = pd.read_csv(file_path, sep=cfg.data.get("separator", ","))
    elif ".tsv" in file_path:
        df = pd.read_csv(file_path, sep="\t")
    else: 
        raise NotImplementedError(f"Only .xlsx, .csv and .tsv formats of datasets are supported, tried {dataset}")

    print("ID column:", cfg.data.id_col)
    print("Group column:", cfg.data.group_col)
    print("Label columns:", cfg.data.label_cols)
    print("Target column:", cfg.base.target_col)

    ############################################
    # Main part start
    ############################################

    #choose of trainer
    if cfg.base.model == "automl":
        trainer = AutomlTrainer(cfg, df)

    elif cfg.base.model == "sklearn":
        trainer = SklearnTrainer(cfg, df)

    elif cfg.base.model == "dummy":
        trainer = DummyTrainer(cfg, df)

    else:
        raise NotImplementedError("Exist only for automl and dummy models")
    
    #main stuff
    if not cfg.base.get("eval_only", False): 
        print("Fitting model...")
        trainer.fit()

    ############################################
    # Main part end
    ############################################
    
    #experiment results aggregation
    experiment_results = ({
        "test": trainer.evaluate(group="test"),
        "train": trainer.evaluate(group="train"),

        "datetime_start": str(datetime_start),
        "datetime_finish": str(datetime.datetime.now()),
        "walltime": str(datetime.datetime.now() - datetime_start),

        "train_shape": trainer.df_train.shape,
        "test_shape": trainer.df_test.shape,
        "train_distribution": "/".join([str(k)+":"+str(v) for k, v in trainer.df_train[cfg.base.target_col].value_counts().items()]),
        "test_distribution": "/".join([str(k)+":"+str(v) for k, v in trainer.df_test[cfg.base.target_col].value_counts().items()]),
    })
    experiment_results.update(OmegaConf.to_container(cfg, resolve=True))
    # if cfg.base.model == "sklearn":
    #     for step in trainer.model.named_steps:
    #         experiment_results.update({step: trainer.model.named_steps[step].get_params()})

    save_experiment_results(flatten_dict(experiment_results), cfg.base.results_path)

    print("------------------------------------------------")
    print("Experiment", cfg.base.experiment_name, "finished")
    print("------------------------------------------------")
    print(json.dumps(
        experiment_results, 
        indent=4, 
        ensure_ascii=False
    ))

    return experiment_results

if __name__ == "__main__":
    run()