import pandas as pd
import numpy as np
from tqdm import tqdm
import os

input_csv_path = 'diagnoses_icd10.csv'

output_dir = 'matrix'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_csv_path)

df_unique = df[['subject_id', 'icd10_category']].drop_duplicates()
unique_icds = df_unique['icd10_category'].dropna().unique()

cat2idx = {icd: i for i, icd in enumerate(unique_icds)}
idx2cat = {i: icd for icd, i in cat2idx.items()}
n = len(unique_icds)

co_occurrence = np.zeros((n, n), dtype=int)

for subject_id, group in tqdm(df_unique.groupby('subject_id')):
    cats = group['icd10_category'].unique()
    indices = [cat2idx[cat] for cat in cats if cat in cat2idx]
    for i in indices:
        for j in indices:
            co_occurrence[i, j] += 1

# patients with each code (diagonal)
patients_per_code = co_occurrence.diagonal().copy()

# co-occurrence (counts)
co_df = pd.DataFrame(co_occurrence, index=unique_icds, columns=unique_icds)
co_df.to_csv(os.path.join(output_dir, "new_cat_co_matrix_counts.csv"))

# fraction  (Jaccard-like - N_ij / (N_i + N_j - N_ij))
denominator = (
    patients_per_code.reshape(-1, 1) +
    patients_per_code.reshape(1, -1) -
    co_occurrence
)
denominator[denominator == 0] = 1  # avoid division by zero
co_occurrence_jaccard = co_occurrence / denominator
fraction_df = pd.DataFrame(co_occurrence_jaccard, index=unique_icds, columns=unique_icds)
fraction_df.to_csv(os.path.join(output_dir, "new_cat_co_matrix_fraction.csv"))

# Conditional probability matrix
patients_per_code_safe = patients_per_code.copy()
patients_per_code_safe[patients_per_code_safe == 0] = 1
conditional_prob = co_occurrence / patients_per_code_safe[:, None]
cond_df = pd.DataFrame(conditional_prob, index=unique_icds, columns=unique_icds)
cond_df.to_csv(os.path.join(output_dir, "new_cat_co_matrix_conditional_prob.csv"))
