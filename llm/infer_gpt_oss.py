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

##########################################

SYSTEM_PROMPT_MULTI_WITH_RANDOM = """I'll give you an INDEX ICD-10 category X (category level only, e.g., C25 — NOT C25.0) and its description.

Task:
For a patient known to have ICD-category X in their medical record, judge for each other category Y (from ICD-10 system) whether Y is:
[A] about as likely to appear as in a random patient;
[B] more likely to appear than in a random patient;
[C] less likely to appear than in a random patient.

Output requirements:
- Return ONLY valid JSON (no extra text).
- Use only ICD category codes from the ICD-10 system (no invented codes, no sub-codes with dots).
- Because [A] would be huge, DO NOT enumerate [A] items. Instead:
  * list only the strongest [B] and [C] categories (those with the clearest deviation from random),
  * omit everything else (treat omitted categories as [A] by default).
- If you are uncertain, omit the category (defaulting it to [A]).

ANSWER IN JSON FORMAT:
{
  "comment": "Brief reasoning: why the listed categories are more/less likely; mention typical clinical links (comorbidity, shared risk factors, complications, diagnostic workup, treatment effects).",
  "answer": {
    "B": <list of categories in square brackets, separated by comma, for example: ["A01", "C05", "H12"]>,  // more likely than random patient
    "C": <list of categories in square brackets, separated by comma, for example: ["A01", "C05", "H12"]>   // less likely than random patient
  }
}"""

TEMPLATE_MULTI_WITH_RANDOM = """{{
    "X": {{
        "icd10_category": {X}, 
        "description": {X_description} 
    }}
}}
"""

##########################################

SYSTEM_PROMPT_MULTI_WITH_RANDOM_AND_REFERENCE = """I'll give you:
1) An INDEX ICD-10 category X (category level only, e.g., C25 — NOT C25.0) and its description.
2) A list of ALLOWED ICD-10 categories available in my dataset, each with a short description.

Task:
For a patient known to have ICD-category X in their medical record, judge for each other category Y (from ALLOWED categories) whether Y is:
[A] about as likely to appear as in a random patient;
[B] more likely to appear than in a random patient;
[C] less likely to appear than in a random patient.

Output requirements:
- Return ONLY valid JSON (no extra text).
- Use only ICD category codes from the ALLOWED list (no invented codes, no sub-codes with dots).
- Because [A] would be huge, DO NOT enumerate [A] items. Instead:
  * list only the strongest [B] and [C] categories (those with the clearest deviation from random),
  * omit everything else (treat omitted categories as [A] by default).
- If you are uncertain, omit the category (defaulting it to [A]).

ANSWER IN JSON FORMAT:
{
  "comment": "Brief reasoning: why the listed categories are more/less likely; mention typical clinical links (comorbidity, shared risk factors, complications, diagnostic workup, treatment effects).",
  "answer": {
    "B": <list of categories in square brackets, separated by comma, for example: ["A01", "C05", "H12"]>,  // more likely than random patient
    "C": <list of categories in square brackets, separated by comma, for example: ["A01", "C05", "H12"]>   // less likely than random patient
  }
}"""

TEMPLATE_MULTI_WITH_RANDOM_AND_REFERENCE = """{{
    "X": {{
        "icd10_category": {X}, 
        "description": {X_description} 
    }},
    "ALLOWED": {ALLOWED}
}}
"""

# "ALLOWED": [
#     {
#         "icd_code": "<Y1>", 
#         "description": "<Y1_description>"
#     },
#     ...
# ]

def get_responses_multi(codes: pd.DataFrame, inferer : GPT_OSS_Inference, log_dir="logs/", n_attempts=10, prompt_with_random=False):
    try: 
        responses = pd.DataFrame(data=["null" for _ in range(len(codes))], index=codes["icd10_category"].values, columns=["response"])

        for _, row in tqdm(codes.iterrows(), total=len(codes)):
            if prompt_with_random:
                query = TEMPLATE_MULTI_WITH_RANDOM.format(
                    X=row["icd10_category"], 
                    X_description=row["description"],
                )

            else:
                query = TEMPLATE_MULTI.format(
                    row["icd10_category"], 
                    row["description"], 
                )

            for i in range(n_attempts):
                try:
                    parsed_response = inferer(query)
                    final_answer = inferer.get_final_answer(parsed_response)
                    responses.loc[row["icd10_category"], "response"] = final_answer
                    break
                    
                except Exception as e:
                    print("Attempt {} for code {}".format(i+1, row["icd10_category"]))
                    print(e)

            if not os.path.exists(log_dir): os.makedirs(log_dir)
            rendered_response = inferer.render_messages(parsed_response)
            with open("{}/{}.txt".format(log_dir, row["icd10_category"]), "w") as f: 
                print(json.dumps({"query": query, "response": rendered_response}, indent=4), file=f)
                
        return responses

    except Exception as e:
        print(e)
        return responses

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
                        help="Maximum number of categories to process. Only for debugging purposes.")
    parser.add_argument("--prompt_with_random", action="store_true", default=False,
                       help="Whether to use prompt with refering to probability of a random patient")
    
    args = parser.parse_args()
    
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
    inferer.set_developer_message(SYSTEM_PROMPT_MULTI_WITH_RANDOM if args.prompt_with_random else SYSTEM_PROMPT_MULTI)
    inferer.generate_args["max_new_tokens"] = 8192

    OUTPUTS_DIR = "outputs"
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)
    responses_save_path = f"{OUTPUTS_DIR}/responses_{model_size}_{reasoning_effort}{'_w_rnd_lvl' if args.prompt_with_random else ''}"
    logs_save_path = f"logs/logs_{model_size}_{reasoning_effort}{'_w_rnd_lvl' if args.prompt_with_random else ''}"
    
    for i in range(args.iterations):
        inferer.set_system_message(reasoning_effort, "2025-06-28")
        print(f"Getting responses from {model_size} with reasoning effort {reasoning_effort} on iteration {i}")
        responses = get_responses_multi(codes, inferer, log_dir=f"{logs_save_path}_{i}/", prompt_with_random=args.prompt_with_random)
        responses.to_csv(f"{responses_save_path}_{i}.tsv", sep="\t")
