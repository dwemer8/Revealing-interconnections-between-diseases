import pandas as pd
import os
from tqdm import tqdm
import json
from gpt_oss_inference_class import GPT_OSS_Inference, GPT_OSS_MODEL_SIZE
from openai_harmony import ReasoningEffort
import torch
import gc
import argparse
import logging

SYSTEM_PROMPT_MCQ = ""

TEMPLATE_MCQ = """ICD-10 category {X} is {X_description}. ICD-10 category {Y} is {Y_description}. Based on this, answer the following multiple-choice question: 
A patient known to have been assigned ICD-category {X} is ... 
[A] likely to have {Y} too; 
[B] not likely to have {Y}. 
Answer with one letter only (A or B)."""

def get_response(inferer, c1, c2, d1, d2, log_dir, n_attempts):
    query = TEMPLATE_MCQ.format(
        X=c1, 
        X_description=d1, 
        Y=c2, 
        Y_description=d2,
    )

    for i in range(n_attempts):
        try:
            parsed_response = inferer(query)
            final_answer = inferer.get_final_answer(parsed_response)
            break
            
        except Exception as e:
            print("Attempt {} for codes {},{}".format(i+1, c1, c2))
            print(e)

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    rendered_response = inferer.render_messages(parsed_response)
    with open("{}/{}_{}.txt".format(log_dir, c1, c2), "w") as f: 
        print(json.dumps({"query": query, "response": rendered_response}, indent=4), file=f)

    return final_answer

def get_responses_mcq(codes: pd.DataFrame, inferer : GPT_OSS_Inference, log_dir="logs/", n_attempts=10):
    try:
        scores = pd.DataFrame()

        for _, row1 in tqdm(codes.iterrows(), total=len(codes)):
            for _, row2 in tqdm(codes.iterrows(), total=len(codes)):
                scores = pd.concat(
                    [
                        scores, 
                        pd.DataFrame.from_records([{
                            "icd10_category_1": row1["icd10_category"],
                            "icd10_category_2": row2["icd10_category"],
                            "response": get_response(
                                inferer,
                                row1["icd10_category"], 
                                row2["icd10_category"], 
                                row1["description"], 
                                row2["description"], 
                                log_dir,
                                n_attempts, 
                            )
                        }])
                    ], 
                    axis="index", 
                    ignore_index=True
                )
                
        return scores

    except Exception as e:
        print(e)
        logging.exception(e)
        return scores
    
def get_responses_mcq_for_single_code(code: str, codes: pd.DataFrame, inferer : GPT_OSS_Inference, log_dir="logs/", n_attempts=10):
    try:
        scores = pd.DataFrame()

        for _, row in tqdm(codes.iterrows(), total=len(codes)):
            scores = pd.concat(
                [
                    scores, 
                    pd.DataFrame.from_records([{
                        "icd10_category_1": code,
                        "icd10_category_2": row["icd10_category"],
                        "response": get_response(
                            inferer,
                            code, 
                            row["icd10_category"], 
                            codes[codes["icd10_category"] == code].iloc[0].loc["description"],
                            row["description"], 
                            log_dir,
                            n_attempts, 
                        )
                    }])
                ], 
                axis="index", 
                ignore_index=True
            )

        for _, row in tqdm(codes.iterrows(), total=len(codes)):
            scores = pd.concat(
                [
                    scores, 
                    pd.DataFrame.from_records([{
                        "icd10_category_1": row["icd10_category"],
                        "icd10_category_2": code,
                        "response": get_response(
                            inferer,                         
                            row["icd10_category"],
                            code,  
                            row["description"], 
                            codes[codes["icd10_category"] == code].iloc[0].loc["description"],
                            log_dir,
                            n_attempts, 
                        )
                    }])
                ], 
                axis="index", 
                ignore_index=True
            )
                
        return scores

    except Exception as e:
        print(e)
        logging.exception(e)
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-OSS inference with configurable model size and reasoning effort")
    parser.add_argument("--model-size", type=str, choices=["SMALL", "BIG"], default="SMALL",
                       help="Model size: SMALL (20B) or BIG (120B)")
    parser.add_argument("--reasoning-effort", type=str, choices=["LOW", "MEDIUM", "HIGH"], default="LOW",
                       help="Reasoning effort level: LOW, MEDIUM, or HIGH")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Number of iterations to run")
    parser.add_argument("--input-csv", type=str, default="icd10_categories_descriptions.csv",
                       help="Path to input CSV file with ICD-10 categories")
    parser.add_argument("--max_categories", type=int, default=None,
                        help="Maximum number of categories to process")
    parser.add_argument("--code", type=str, default=None, 
                        help="Code to process ('R18', for example); if not specified, all codes will be processed")
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("CUDA is available")
    else: 
        print("CUDA is not available")
    
    print(torch.cuda.memory_allocated()/1024**3, "GB")
    gc.collect()          # run Python garbage collector
    torch.cuda.empty_cache()  # free cached memory to the OS
    print(torch.cuda.memory_allocated()/1024**3, "GB")

    codes = pd.read_csv(args.input_csv).drop("Unnamed: 0", axis=1, errors="ignore")
    if args.max_categories is not None:
        codes = codes[:args.max_categories]

    # Convert string arguments to enum values
    model_size = GPT_OSS_MODEL_SIZE[args.model_size]
    reasoning_effort = ReasoningEffort[args.reasoning_effort]

    inferer = GPT_OSS_Inference(model_size)
    inferer.set_developer_message(SYSTEM_PROMPT_MCQ)
    inferer.generate_args["max_new_tokens"] = 8192

    OUTPUTS_DIR = "outputs_mcq"
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)
    
    for i in range(args.iterations):
        inferer.set_system_message(reasoning_effort, "2025-06-28")
        print(f"Getting responses from {model_size} with reasoning effort {reasoning_effort} on iteration {i}")
        if args.code is None:
            responses = get_responses_mcq(codes, inferer, log_dir=f"logs_mcq/logs_{model_size}_{reasoning_effort}_{i}/")
            responses.to_csv(f"{OUTPUTS_DIR}/responses_{model_size}_{reasoning_effort}_{i}.tsv", sep="\t")
        else:
            responses = get_responses_mcq_for_single_code(args.code, codes, inferer, log_dir=f"logs_mcq/logs_{model_size}_{reasoning_effort}_{i}/")
            responses.to_csv(f"{OUTPUTS_DIR}/responses_{model_size}_{reasoning_effort}_{args.code}_{i}.tsv", sep="\t")

        