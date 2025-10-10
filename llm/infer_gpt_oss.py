import pandas as pd
import os
from tqdm import tqdm
import json
from gpt_oss_inference_class import GPT_OSS_Inference, GPT_OSS_MODEL_SIZE
from openai_harmony import ReasoningEffort
import torch
import gc
import argparse

SYSTEM_PROMPT_MULTI = """I'll give you ICD-10 categories (for example, C25, NOT C25.0!) and thier descriptions. You have to tell me, If a patient has an ICD code for a given category in their medical record, what other categories of codes are also likely to be in their medical record?

ANSWER IN JSON FORMAT:
{
    "comment": <your thoughts and explanations>,
    "answer": <list of categories in square brackets, separated by comma, for example: [A01, C05, ..., H12]>
}
DO NOT ADD ANYTHING ELSE IN YOUR ANSWER."""

TEMPLATE_MULTI = """{{
    icd_code: {},
    description: {},
}}"""

def get_responses_multi(codes: pd.DataFrame, inferer : GPT_OSS_Inference, log_dir="logs/", n_attempts=10):
    try:
        scores = pd.DataFrame(data=["null" for _ in range(len(codes))], index=codes["icd10_category"].values, columns=["response"])

        for _, row in tqdm(codes.iterrows(), total=len(codes)):
            query = TEMPLATE_MULTI.format(
                row["icd10_category"], 
                row["description"], 
            )

            for i in range(n_attempts):
                try:
                    parsed_response = inferer(query)
                    final_answer = inferer.get_final_answer(parsed_response)
                    scores.loc[row["icd10_category"], "response"] = final_answer
                    break
                    
                except Exception as e:
                    print("Attempt {} for code {}".format(i+1, row["icd10_category"]))
                    print(e)

            if not os.path.exists(log_dir): os.makedirs(log_dir)
            rendered_response = inferer.render_messages(parsed_response)
            with open("{}/{}.txt".format(log_dir, row["icd10_category"]), "w") as f: print(json.dumps({"query": query, "response": rendered_response}, indent=4), file=f)
                
        return scores

    except Exception as e:
        print(e)
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-OSS inference with configurable model size and reasoning effort")
    parser.add_argument("--model-size", type=str, choices=["SMALL", "BIG"], default="SMALL",
                       help="Model size: SMALL (20B) or BIG (120B)")
    parser.add_argument("--reasoning-effort", type=str, choices=["LOW", "MEDIUM", "HIGH"], default="MEDIUM",
                       help="Reasoning effort level: LOW, MEDIUM, or HIGH")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Number of iterations to run")
    parser.add_argument("--input-csv", type=str, default="icd10_categories_descriptions.csv",
                       help="Path to input CSV file with ICD-10 categories")
    
    args = parser.parse_args()
    
    print(torch.cuda.memory_allocated()/1024**3, "GB")
    gc.collect()          # run Python garbage collector
    torch.cuda.empty_cache()  # free cached memory to the OS
    print(torch.cuda.memory_allocated()/1024**3, "GB")

    codes = pd.read_csv(args.input_csv).drop("Unnamed: 0", axis=1, errors="ignore")

    # Convert string arguments to enum values
    model_size = GPT_OSS_MODEL_SIZE[args.model_size]
    reasoning_effort = ReasoningEffort[args.reasoning_effort]

    inferer = GPT_OSS_Inference(model_size)
    inferer.set_developer_message(SYSTEM_PROMPT_MULTI)
    inferer.generate_args["max_new_tokens"] = 8192

    OUTPUTS_DIR = "outputs"
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)
    
    for i in range(args.iterations):
        inferer.set_system_message(reasoning_effort, "2025-06-28")
        print(f"Getting responses from {model_size} with reasoning effort {reasoning_effort} on iteration {i}")
        responses = get_responses_multi(codes, inferer, log_dir=f"logs/logs_{model_size}_{reasoning_effort}_{i}/")
        responses.to_csv(f"{OUTPUTS_DIR}/responses_{model_size}_{reasoning_effort}_{i}.tsv", sep="\t")
