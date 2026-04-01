import string
import pandas as pd

# MKB codes
def is_mkb(v):
    return len(v) >= 3 and (
        v[0] in string.ascii_uppercase or (ord(v[0])>=ord('А') and ord(v[0])<=ord('Я'))
    ) and v[1].isdigit() and v[2].isdigit()\
    and "-" not in v #ICD blocks

def is_cancer_mkb(v):
    return is_mkb(v) and (ord(v[0]) in [ord('С'), ord('C')] or (ord(v[0]) == ord('D') and 0 <= int(v[1:3]) <= 48))

def get_history_before_event(patient_df:pd.DataFrame, target:str="C34"):
    if target not in patient_df["icd10_category"].values:
        return None
    else:
        patient_df = patient_df.sort_values(["admittime", "dischtime", "seq_num"]).reset_index(drop=True)
        target_index = patient_df[patient_df["icd10_category"] == target].index[0]
        target_admission = patient_df[patient_df["icd10_category"] == target].iloc[0]["hadm_id"]
        prev_patient_df = patient_df.loc[:target_index]
        prev_patient_df = prev_patient_df[prev_patient_df["hadm_id"] != target_admission]
        return prev_patient_df
    
def is_target_the_first_cancer(patient_df:pd.DataFrame, target:str="C34"):
    prev_patient_df = get_history_before_event(patient_df, target=target)
    if prev_patient_df is not None:
        return not prev_patient_df["icd10_category"].apply(is_cancer_mkb).any()
    else:
        return True

def n_admissions_before_event(patient_df:pd.DataFrame, target:str="C34"):
    prev_patient_df = get_history_before_event(patient_df, target=target)
    if prev_patient_df is not None:
        return prev_patient_df["hadm_id"].nunique()
    else:
        return None
    
def n_codes_before_event(patient_df:pd.DataFrame, target:str="C34"):
    prev_patient_df = get_history_before_event(patient_df, target=target)
    if prev_patient_df is not None:
        return prev_patient_df["icd10_code"].nunique()
    else:
        return None
    
def timedelta_to_event_from_first_record(patient_df:pd.DataFrame, target:str="C34"):
    if target not in patient_df["icd10_category"].values:
        return None
    else:
        patient_df = patient_df.sort_values(["admittime", "dischtime", "seq_num"]).reset_index(drop=True)
        target_data = patient_df[patient_df["icd10_category"] == target].iloc[0]["admittime"]
        first_data = patient_df.iloc[0]["admittime"]
        return (target_data - first_data).days
    
def timedelta_to_event_from_prev_record(patient_df:pd.DataFrame, target:str="C34"):
    if target not in patient_df["icd10_category"].values:
        return None
    else:
        patient_df = patient_df.sort_values(["admittime", "dischtime", "seq_num"]).reset_index(drop=True)
        target_data = patient_df[patient_df["icd10_category"] == target].iloc[0]["admittime"]
        prev_patient_df = get_history_before_event(patient_df, target=target)
        if prev_patient_df is not None:
            first_data = prev_patient_df.iloc[-1]["admittime"]
            return (target_data - first_data).days
        else:
            return None
        
def last_timedelta(patient_df:pd.DataFrame):
    admittimes = sorted(patient_df["admittime"].unique())
    if len(admittimes) < 2:
        return None
    else:
        return (admittimes[-1] - admittimes[-2]).days