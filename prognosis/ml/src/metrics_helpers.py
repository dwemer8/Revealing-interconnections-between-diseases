import scipy.stats as sts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Any, List, Tuple, Union
from functools import partial

from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    balanced_accuracy_score
)

def specificity_score(
        y_test: np.array, 
        y_prob: np.array, 
        threshold: float = 0.5,
        zero_division: Union[int, float, str] = "warn"
    ):
    cf_mtrx = confusion_matrix(y_test, y_prob > threshold)
    if cf_mtrx[0,0]+cf_mtrx[0,1] == 0:
        if zero_division == "warn":
            specificity = 0
            print("Specificity is zero because there are no positives in test set.")
        elif zero_division == 0:
            specificity = 0
        elif zero_division == 1:
            specificity = 1
        elif zero_division == np.nan:
            specificity = np.nan
        else:
            raise ValueError("zero_division should be one of ['warn', 0, 1, np.nan]")
    else:
        specificity = cf_mtrx[0,0]/(cf_mtrx[0,0]+cf_mtrx[0,1])
    return specificity

def sensitivity_score(
        y_test: np.array, 
        y_prob: np.array, 
        threshold: float = 0.5,
        zero_division: Union[int, float, str] = "warn"
    ):
    cf_mtrx = confusion_matrix(y_test, y_prob > threshold)
    if cf_mtrx[0,0]+cf_mtrx[0,1] == 0:
        if zero_division == "warn":
            sensitivity = 0
            print("Specificity is zero because there are no positives in test set.")
        elif zero_division == 0:
            sensitivity = 0
        elif zero_division == 1:
            sensitivity = 1
        elif zero_division == np.nan:
            sensitivity = np.nan
        else:
            raise ValueError("zero_division should be one of ['warn', 0, 1, np.nan]")
    else:
        sensitivity = cf_mtrx[1,1]/(cf_mtrx[1,0]+cf_mtrx[1,1])
    return sensitivity

def compute_bootstrapped_score(
        y_test: np.array, 
        y_prob: np.array, 
        scorer: Callable[[np.array, np.array], float], 
        m_sample: int = None, 
        stratum_vals: np.array =None
    ):
    assert not isinstance(y_test, pd.Series), "y_test should be np.array"
    assert not isinstance(y_prob, pd.Series), "y_prob should be np.array"

    idx = np.array(range(len(y_test)))
    if m_sample is None: m_sample = len(y_test) #bootstrap sample size
        
    if stratum_vals is not None: #select equal number of samples from each category
        idx_bs = [] 
        for val in set(stratum_vals):
            stratum_idx = idx[stratum_vals == val] 
            idx_bs += np.random.choice(stratum_idx, size=len(stratum_idx), replace=True).tolist()
    else:
        idx_bs = np.random.choice(idx, size=m_sample, replace=True)

    try:
        return scorer(y_test[idx_bs], y_prob[idx_bs])
    except Exception as e:
        print("WARNING: Bootstrapping failed for", scorer.func.__name__ if isinstance(scorer, partial) else scorer.__name__, "with error", e)
        return np.nan
    
def compute_ci(
    y_test: np.array, 
    y_prob: np.array, 
    stratum_vals: np.array = None, 
    n_bootstraps: int = 1000, 
    m_sample: int = None, 
    scorer: Callable[[np.array, np.array], float] = average_precision_score,
    alpha: float = 0.05, #95% CI
    verbose: int = 1, 
    return_se: bool = False
):
    assert len(y_test) == len(y_prob), "y_test and y_prob should have the same lengths"
        
    scores = []
    if verbose > 0:
        print(f"Bootstrap scores computing for {scorer.func.__name__ if isinstance(scorer, partial) else scorer.__name__}...")
        for _ in tqdm(range(n_bootstraps)):
            scores.append(compute_bootstrapped_score(y_test, y_prob, scorer, stratum_vals=stratum_vals, m_sample=m_sample))
    else:
        for _ in range(n_bootstraps):
            scores.append(compute_bootstrapped_score(y_test, y_prob, scorer, stratum_vals=stratum_vals, m_sample=m_sample))
    scores = np.array(scores)
    
    nans_share = np.sum(np.isnan(scores).astype(int))/len(scores)
    if nans_share > 0.5: #empirical threshold, you can change it if you have better solution
        print(f"WARNING: There is {nans_share*100:.0f}% NaNs in bootstrapped scores for {scorer.func.__name__ if isinstance(scorer, partial) else scorer.__name__}")
        random_idxs = np.random.choice(list(range(len(y_test))), size=20, replace=False)
        print("       random 10 entries from y_test:", y_test[random_idxs])
        print("corresponging 10 entries from y_prob:", y_prob[random_idxs])
        if return_se:
            return np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan
    
    estimation = np.nanmean(scores)
    se = np.nanstd(scores)
    perc = sts.norm.ppf(1 - alpha/2)
    e_perc = se * perc
    
    if verbose > 1:
        plt.figure(figsize=(4, 2.5))
        plt.hist(scores, bins=50)
        plt.axvline(x = estimation, color = 'tab:orange', label = 'mean')
        plt.axvline(x = estimation - e_perc, color = 'tab:red', label = f'mean - e_{1-alpha:.2f}')
        plt.axvline(x = estimation + e_perc, color = 'tab:red', label = f'mean + e_{1-alpha:.2f}')
        plt.show()
    
    if return_se:
        return estimation, e_perc, se
    else:
        return estimation, e_perc

def compute_scores(
    y_test: np.array,
    y_prob: np.array,
    scorer: Callable[[np.array, np.array], float],
    n_thresholds: int = 200,
    verbose: int = 1
):
    scores = []
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    if verbose > 0:
        print("Checking thresholds...")
    for threshold in tqdm(thresholds):
        scores.append(scorer(y_test, y_prob > threshold))
    return np.array(scores), thresholds

def find_optimal_threshold(
    y_test: np.array, 
    y_prob: np.array, 
    scorer: Callable[[np.array, np.array], float], 
    n_thresholds: int = 200,
    is_max: bool = True,
    verbose: int = 1
):
    assert not isinstance(y_test, pd.Series), "y_test should be np.array"
    assert not isinstance(y_prob, pd.Series), "y_prob should be np.array"
    
    scores, thresholds = compute_scores(y_test, y_prob, scorer, n_thresholds=n_thresholds, verbose=verbose)
    if np.all(np.isnan(scores)):
        return None
    else:
        threshold = thresholds[np.nanargmax(scores) if is_max else np.nanargmin(scores)]
        return threshold

def find_threshold_for_given_metric_value(
    y_test: np.array,
    y_prob: np.array,
    scorer: Callable[[np.array, np.array], float],
    metric_value: float,
    n_thresholds: int = 200,
    verbose: int = 1
):
    assert not isinstance(y_test, pd.Series), "y_test should be np.array"
    assert not isinstance(y_prob, pd.Series), "y_prob should be np.array"

    scores, thresholds = compute_scores(y_test, y_prob, scorer, n_thresholds=n_thresholds, verbose=verbose)
    return thresholds[np.argmin(np.abs(scores - metric_value))]

def get_scorer_name(scorer: Callable[[np.array, np.array], float]):
    return scorer.func.__name__.replace("_score", "") if isinstance(scorer, partial) else scorer.__name__.replace("_score", "")

def evaluate_metrics(
    y_test: np.array, 
    y_pred_proba: np.array,
    scorers: List[Tuple[Callable[[np.array, np.array], float], str]] = [
        (roc_auc_score, "soft"), 
        (average_precision_score, "soft"), 
        (accuracy_score, "hard"),
        (balanced_accuracy_score, "hard"),
        (partial(f1_score, zero_division=np.nan), "hard"), 
        (partial(precision_score, zero_division=np.nan), "hard"), 
        (partial(recall_score, zero_division=np.nan), "hard"), 
        (partial(sensitivity_score, zero_division=np.nan), "hard"), 
        (partial(specificity_score, zero_division=np.nan), "hard")
    ],
    threshold: Union[float, str] = "auto", 
    ci: bool = False,
    verbose: int = 1,
    **kwargs
):
    assert isinstance(threshold, float) or threshold == "auto", "threshold should be float or 'auto'"

    metrics = {}
    for scorer, metric_type in scorers:
        key = get_scorer_name(scorer)
        
        if metric_type == "soft":
            y_pred = y_pred_proba

        elif metric_type == "hard":
            if threshold == "auto":
                if get_scorer_name(scorer) in ["f1", "accuracy", "balanced_accuracy"]:
                    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, scorer, verbose=verbose)
                    key += "_max"

                elif get_scorer_name(scorer) in ["precision", "recall"]:
                    optimal_threshold = find_threshold_for_given_metric_value(y_test, y_pred_proba, recall_score, metric_value=0.90, verbose=verbose)
                    key += "_re90"

                elif get_scorer_name(scorer) in ["sensitivity", "specificity"]:
                    optimal_threshold = find_threshold_for_given_metric_value(y_test, y_pred_proba, specificity_score, metric_value=0.90, verbose=verbose)
                    key += "_sp90"

                else:
                    print(f"WARNING: threshold is set to 0.5 because there is not realization of optimal threshold for {get_scorer_name(scorer)}.")
                    optimal_threshold = 0.5

            else:
                optimal_threshold = threshold

            if optimal_threshold is None:
                print(f"ERROR: all thresholds for {get_scorer_name(scorer)} give NaN, skipping...")
                continue
            
            y_pred = (y_pred_proba > optimal_threshold).astype(int)
            metrics[key + ".threshold"] = optimal_threshold

        else:
            raise NotImplementedError("Exist only for soft and hard metrics")
        
        if ci:
            m, e_95, se = compute_ci(y_test, y_pred, scorer=scorer, return_se=True, verbose=verbose, **kwargs) 
            metrics[key] = m
            metrics[key+".se"] = se
            metrics[key+".e_95"] = e_95
        else:
            metrics[key] = scorer(y_test, y_pred)
    return metrics
