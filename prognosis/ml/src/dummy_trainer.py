import numpy as np

from src.trainer import Trainer
from src.metrics_helpers import evaluate_metrics

class DummyTrainer(Trainer):
    def evaluate(self, group="test"):
        if group == "test":
            y_true = self.df_test[self.cfg.base.target_col]
        elif group == "train":
            y_true = self.df_train[self.cfg.base.target_col]
        else:
            raise ValueError("Only test and train groups are supported")

        if self.cfg.model.get("strategy", "always_true") == "always_true":
            y_pred_proba = np.ones_like(y_true)
            
        elif self.cfg.model.get("strategy", "largest_class") == "largest_class":
            if self.df_test[self.cfg.base.target_col].value_counts().sort_values(ascending=False).index[0] == 0:
                y_pred_proba = np.zeros_like(y_true)
            else:
                y_pred_proba = np.ones_like(y_true)
        
        else:
            raise NotImplementedError("Exist only for always_true and largest_class strategies")

        return super().evaluate(y_pred_proba, group=group)