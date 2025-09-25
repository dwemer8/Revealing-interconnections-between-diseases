import importlib
import sklearn
from sklearn.pipeline import Pipeline

def get_pipeline(stages, stages_kwargs={}):
    stages_imported = []
    for k, v in stages.items():
        module = importlib.import_module("sklearn." + ".".join(v.split(".")[:-1]))
        stages_imported.append((
            k, 
            getattr(module, v.split(".")[-1])(**stages_kwargs.get(k, {}))
        ))
    return Pipeline(stages_imported)

def get_pipeline_stages_kwargs(best_parameters: dict):
    stages_kwargs = {}
    for k, v in best_parameters.items():
        stage = k.split("__")[0]
        param = k.split("__")[1]
        if stage not in stages_kwargs:
            stages_kwargs[stage] = {param: v}
        else:
            stages_kwargs[stage][param] = v

    return stages_kwargs