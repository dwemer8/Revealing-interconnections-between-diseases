import os
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import joblib

from src.metrics_helpers import evaluate_metrics
from src.utils import check_tags

MODELS_SAVE_DIR = "models"

class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        df: pd.DataFrame
    ):
        self.cfg = cfg
        np.random.seed(cfg.base.random_state)
        self.initialize_datasets(cfg.data, df, cfg.base.target_col)

    def initialize_datasets(
        self, 
        cfg: DictConfig, #data config
        df: pd.DataFrame,
        target_col: str
    ):
        self.get_feature_col_names(cfg, df)
        if self.cfg.get("dummy_cols", False):
            df = self.get_dummy_cols(cfg, df)

        self.df_train = df[df[cfg.group_col] == "train"].drop(columns=[cfg.group_col])
        self.df_test = df[df[cfg.group_col] == "test"].drop(columns=[cfg.group_col])

        print("Train shape:", self.df_train[self.feature_cols].shape)
        print("Test shape:", self.df_test[self.feature_cols].shape)
        print("Train classes distribution:\n", self.df_train[target_col].value_counts())
        print("Test classes distribution:\n", self.df_test[target_col].value_counts())

    def get_dummy_cols(
        self,
        cfg: DictConfig, #data config
        df: pd.DataFrame,
    ):
        df = pd.get_dummies(df, dummy_na=True, columns=self.categorical_cols)

        categorical_cols_new = []
        for new_col in list(df.columns):
            for col in self.categorical_cols:
                if new_col.startswith(col):
                    categorical_cols_new.append(new_col)
                    break
        self.feature_cols = [*self.numerical_cols, *categorical_cols_new] #ADDING NEW COLS!

    def get_feature_col_names(
        self,
        cfg: DictConfig, #data config
        df: pd.DataFrame,
    ):
        self.feature_cols = [
            col for col in list(df.columns) 
            if col not in [*cfg.label_cols, cfg.id_col, cfg.group_col]
        ]
        if cfg.get("feature_cols", None) is not None:
            whitelist = cfg.feature_cols.get("whitelist_tags", [])
            blacklist = cfg.feature_cols.get("blacklist_tags", [])
            self.feature_cols = [
                col for col in self.feature_cols 
                if (check_tags(col, whitelist) or len(whitelist) == 0) and\
                not check_tags(col, blacklist)
            ]
        if cfg.get("categorical_cols", None) is not None:
            whitelist = cfg.categorical_cols.get("whitelist_tags", [])
            blacklist = cfg.categorical_cols.get("blacklist_tags", [])
            self.categorical_cols = [
                col for col in self.feature_cols 
                if (check_tags(col, whitelist) or len(whitelist) == 0) and\
                not check_tags(col, blacklist)
            ]
        else:
            self.categorical_cols = []
        self.numerical_cols = [col for col in self.feature_cols if col not in self.categorical_cols]

        if len(self.categorical_cols) <= 40:
            print("Categorical columns:", self.categorical_cols)
        else:
            print("Categorical columns:", self.categorical_cols[:20], "...", self.categorical_cols[-20:])
        if len(self.numerical_cols) <= 40:
            print("Numerical columns:", self.numerical_cols)
        else:
            print("Numerical columns:", self.numerical_cols[:20], "...", self.numerical_cols[-20:])

    def fit(self):
        pass

    def evaluate(self, y_pred_proba, group="test"):
        if group == "test":
            y_true = self.df_test[self.cfg.base.target_col].values
        elif group == "train":
            y_true = self.df_train[self.cfg.base.target_col].values
        else:
            raise ValueError("Only test and train groups are supported")
        
        return evaluate_metrics(
            y_true, 
            y_pred_proba, 
            ci=self.cfg.base.get("ci", False), 
            n_bootstraps=self.cfg.base.n_bootstraps if self.cfg.base.get("n_bootstraps", False) else 1000,
            threshold=self.cfg.base.get("threshold", "auto")
        )

    def save_model(self):
        if not os.path.exists(MODELS_SAVE_DIR): os.makedirs(MODELS_SAVE_DIR)
        joblib.dump(self.model, f'{MODELS_SAVE_DIR}/{self.cfg.base.experiment_name}_{self.cfg.data.nosology}_{self.cfg.base.target_col}.pkl')

    def load_model(self):
        return joblib.load(f'{MODELS_SAVE_DIR}/{self.cfg.model.pretrained_model}.pkl')