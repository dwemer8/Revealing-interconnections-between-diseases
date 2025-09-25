import os
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import joblib
import json
from copy import deepcopy
from typing import Union, Dict, List
from tqdm import tqdm, trange

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from src.metrics_helpers import evaluate_metrics
from src.trainer import Trainer
from src.sklearn_parse_helpers import get_pipeline, get_pipeline_stages_kwargs
from src.type_converter import get_type_conversion_function

class SklearnTrainer(Trainer):
    def __init__(
        self,
        cfg : DictConfig,
        df : pd.DataFrame
    ):
        super().__init__(cfg, df)
        self.initialize_model(cfg)

    def initialize_model(
        self,
        cfg: DictConfig,
        param_grid: Union[Dict, List] = None,
    ):
        if cfg.model.get("is_load_pretrained", False):
            self.model = self.load_model()

        else:
            scorer = getattr(sklearn.metrics, cfg.model.scorer)
            if param_grid is None: #do not save param_grid_extrapolation
                self.param_grid = OmegaConf.to_container(cfg.model.grid_search_params)

            self.gscv = GridSearchCV(
                get_pipeline(cfg.model.pipeline),
                param_grid=self.param_grid if param_grid is None else param_grid,
                scoring=make_scorer(scorer, needs_proba=True if cfg.model.scorer_type == "soft" else False),
                cv=StratifiedKFold(n_splits=cfg.base.n_folds, shuffle=True, random_state=cfg.base.random_state),
                n_jobs=self.cfg.model.get("n_jobs", -1),
                verbose=max(cfg.base.get("verbose", 1), 0)
            )


    def fit(self):
        X_train = self.df_train[self.feature_cols].fillna(-1) #REMOVE
        y_train = self.df_train[self.cfg.base.target_col]
        self.best_parameters = self.find_best_parameters(X_train, y_train)
        self.model = get_pipeline(self.cfg.model.pipeline, get_pipeline_stages_kwargs(self.best_parameters))
        self.model.fit(X_train, y_train)
        self.save_model()


    def find_best_parameters(
        self, 
        X_train, 
        y_train
    ):
        self.cv_results = pd.DataFrame() 
        best_parameters = None
        param_grid_extrapolation = []
        max_attempts = self.cfg.model.get("gs_extrapolation_max_attempts", 0)
        for i in trange(max_attempts + 1):
            self.initialize_model(self.cfg, param_grid=param_grid_extrapolation if i > 0 else None)
            self.gscv.fit(X_train, y_train)

            self.cv_results = pd.concat([self.cv_results, pd.DataFrame(self.gscv.cv_results_)], axis=0, ignore_index=True)
            best_parameters = self.cv_results.loc[self.cv_results["mean_test_score"].idxmax(), "params"]

            param_grid_extrapolation, param_grid_extension = self.get_param_grid_for_search_extrapolation(best_parameters)
            if len(param_grid_extrapolation) == 0:
                tqdm.write(f"Best parameters have been found: {best_parameters}")
                break
            else:
                tqdm.write(
                    f"Parameters with values at the edge of the grid have been found, new values will be tried:\
                    {json.dumps(param_grid_extension, indent=4, ensure_ascii=False)}", 
                )

            if i == max_attempts:
                tqdm.write(f"Max number of attempts {max_attempts} is reached")

        param_cols = [col for col in self.cv_results.columns if col.startswith("param_")]
        print(self.cv_results.sort_values("mean_test_score", ascending=False)[["mean_test_score", *param_cols]])
        log_path = self.cfg.base.get("log_path", "logs")
        if not os.path.exists(log_path): os.makedirs(log_path)
        self.cv_results.to_csv(f"{log_path}/{self.cfg.base.experiment_name}_{self.cfg.data.nosology}_{self.cfg.base.target_col}_cv_results.csv")

        return best_parameters
    

    def get_param_grid_for_search_extrapolation(self, best_parameters):
        param_grid_extension = self.get_param_grid_extension(
            best_parameters, 
            self.param_grid, 
            self.cfg.model.get("gs_extrapolation_params", None)
        )        
        self.param_grid = self.update_param_grid(self.param_grid, param_grid_extension)
        
        param_grid_extrapolation = [] 
        for param in param_grid_extension:
            tmp = deepcopy(self.param_grid)
            tmp[param] = param_grid_extension[param]
            param_grid_extrapolation.append(tmp)
        return param_grid_extrapolation, param_grid_extension
            

    def update_param_grid(self, param_grid, param_grid_extension):
        param_grid = deepcopy(param_grid)
        for param in param_grid_extension:
            param_grid[param].extend(param_grid_extension[param])
            param_grid[param].sort() #to find parameters on the edges further
        return param_grid


    def get_param_grid_extension(self, best_parameters, param_grid, extrapolation_params):
        if extrapolation_params is None: return []
        elif isinstance(extrapolation_params, DictConfig): 
            extrapolation_params = OmegaConf.to_container(extrapolation_params)

        param_grid_extension = {}
        for param, direction, p_value in self.get_parameters_on_the_edge(
            best_parameters,
            param_grid,
            extrapolation_params
        ):
            if len(param_grid[param]) >= extrapolation_params[param].get("max_entries", np.inf):
                print(f"Maximum number of entries {len(param_grid[param])} for parameter {param} is reached")
                continue

            step = extrapolation_params[param].get("step", 10)
            min_value = extrapolation_params[param].get("min", -np.inf)
            max_value = extrapolation_params[param].get("max", np.inf)
            to_type = get_type_conversion_function(extrapolation_params[param].get("type", str(type(p_value).__name__)))
            if extrapolation_params[param]["scale"] == "log":
                if direction == "minmax":
                    new_values = []
                    if to_type(p_value / step) >= min_value:
                        new_values.append(to_type(p_value / step))
                    if to_type(p_value * step) <= max_value:
                        new_values.append(to_type(p_value * step))
                    if len(new_values) > 0:
                        param_grid_extension[param]  = new_values

                elif direction == "min":
                    if to_type(p_value / step) >= min_value:
                        param_grid_extension[param]  = [to_type(p_value / step)]

                elif direction == "max":
                    if to_type(p_value * step) <= max_value:
                        param_grid_extension[param]  = [to_type(p_value * step)]

                else:
                    raise NotImplementedError("Only min, max and minmax directions are supported")

            elif extrapolation_params[param]["scale"] == "linear":
                if direction == "minmax":
                    new_values = []
                    if to_type(p_value - step) >= min_value:
                        new_values.append(to_type(p_value - step))
                    if to_type(p_value + step) <= max_value:
                        new_values.append(to_type(p_value + step))
                    if len(new_values) > 0:
                        param_grid_extension[param]  = new_values

                elif direction == "min":
                    if to_type(p_value - step) >= min_value:
                        param_grid_extension[param]  = [to_type(p_value - step)]

                elif direction == "max":
                    if to_type(p_value + step) <= max_value:
                        param_grid_extension[param]  = [to_type(p_value + step)]

                else:
                    raise NotImplementedError("Only min, max and minmax directions are supported")
                    
            else:
                raise NotImplementedError("Only log and linear scales are supported")
        return param_grid_extension
    

    def get_parameters_on_the_edge(self, best_parameters, param_grid, extrapolation_params=None):
        parameters_on_the_edge = []
        # print("Best parameters:", best_parameters)
        # print("Param grid:", param_grid)
        # print("Extrapolation params:", extrapolation_params)
        for param in extrapolation_params:
            p_value = best_parameters[param]
            p_index = param_grid[param].index(p_value)
            if p_index in [0, len(param_grid[param])-1]: #if param is at the edge of the grid
                if len(param_grid[param]) == 1:
                    direction = "minmax"
                elif p_index == 0:
                    direction = "min"
                else:
                    direction = "max"
                parameters_on_the_edge.append((param, direction, p_value))
        return parameters_on_the_edge
    

    def evaluate(self, group="test"):
        if group == "test":
            X_test = self.df_test[self.feature_cols].fillna(-1) #REMOVE
            y_pred_proba = self.model.predict(X_test)#[:, 1]
        elif group == "train":
            X_train = self.df_train[self.feature_cols].fillna(-1) #REMOVE
            y_pred_proba = self.model.predict(X_train)#[:, 1]
        else:
            raise ValueError("Only test and train groups are supported")
        
        return super().evaluate(y_pred_proba, group=group)

    