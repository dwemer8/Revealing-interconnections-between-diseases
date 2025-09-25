import os
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
from sklearn.model_selection import StratifiedGroupKFold
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco, ReportDecoUtilized

from src.trainer import Trainer

REPORTS_DIR = "lama_reports"

class AutomlTrainer(Trainer):
    def __init__(
        self,
        cfg : DictConfig,
        df : pd.DataFrame
    ):
        super().__init__(cfg, df)
        torch.set_num_threads(cfg.model.reader_params.n_jobs)
        self.initialize_model(cfg)

    def initialize_model(
        self,
        cfg: DictConfig,
    ):
        if cfg.model.get("is_load_pretrained", False):
            self.model = self.load_model()

        else:
            scorer = getattr(sklearn.metrics, cfg.model.get("scorer", "roc_auc_score"))

            self.model = TabularAutoML(
                task = Task(name = 'binary', metric = scorer),
                timeout = cfg.model.get("timeout", None),
                cpu_limit = cfg.model.get("cpu_limit", None),
                memory_limit = cfg.model.get("memory_limit", None),
                gpu_ids = cfg.model.get("gpu_ids", None),
                reader_params = {
                    'n_jobs': cfg.model.reader_params.n_jobs, 
                    'cv': cfg.base.n_folds, 
                    'random_state': cfg.base.random_state
                },
                tuning_params = cfg.model.get("tuning_params", {}),
                general_params = cfg.model.get("general_params", {}),
                selection_params = cfg.model.get("selection_params", {}),
                debug = cfg.model.get("debug", False),
            )

        if not os.path.exists(REPORTS_DIR): os.makedirs(REPORTS_DIR)
        RD = ReportDeco(output_path = f'{REPORTS_DIR}/{self.cfg.base.experiment_name}_{self.cfg.data.nosology}_{self.cfg.base.target_col}')
        self.model_rd = RD(self.model)

    def fit(self):
        self.df_train = self.df_train.reset_index(drop=True)
        y = self.df_train[self.cfg.base.target_col].to_numpy()
        groups = self.df_train[self.cfg.data.id_col].to_numpy()
        idx = np.arange(len(self.df_train))

        skf = StratifiedGroupKFold(
            n_splits=self.cfg.base.n_folds,
            shuffle=True,
            random_state=self.cfg.base.random_state
        )
        folds = list(skf.split(idx, y=y, groups=groups))

        # other_labels = self.cfg.data.label_cols.copy()
        # other_labels.remove(self.cfg.base.target_col)

        oof_preds = self.model_rd.fit_predict(
            self.df_train[[*self.feature_cols, self.cfg.base.target_col]], 
            roles = {
                "target": self.cfg.base.target_col,
                "numeric": self.numerical_cols,
                "category": self.categorical_cols,
                # "drop": [*other_labels, self.cfg.data.id_col],
                # "group": self.cfg.data.id_col, #it will use GroupKFold, not StratifiedGroupKFold
            },
            verbose=4,
            cv_iter=folds
        ).data

        print(self.model.create_model_str_desc())
        self.save_model()
        return oof_preds
    
    def evaluate(self, group="test"):
        if group == "test":
            y_pred_proba = self.model_rd.predict(self.df_test[[*self.feature_cols, self.cfg.base.target_col]]).data[:, 0]
        elif group == "train":
            y_pred_proba = self.model_rd.predict(self.df_train[[*self.feature_cols, self.cfg.base.target_col]]).data[:, 0]
            
        return super().evaluate(y_pred_proba, group=group)
