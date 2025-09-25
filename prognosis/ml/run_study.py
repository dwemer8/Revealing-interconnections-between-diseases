import hydra
from omegaconf import DictConfig, open_dict

from run import run

@hydra.main(version_base=None, config_path="configs", config_name="run_study")
def run_study(cfg : DictConfig) -> None:
    return run(cfg)["test"]["roc_auc"]

if __name__ == "__main__":
    run_study()

        

        
        
        
        
        

        