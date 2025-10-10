import pandas as pd
import os
from tqdm import tqdm
import json
import torch
import gc
import argparse
from datetime import datetime, timedelta
import torch

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

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

def get_responses_multi(codes: pd.DataFrame, model_id: str, log_dir="logs/", n_attempts=10):
    try:
        scores = pd.DataFrame(data=["null" for _ in range(len(codes))], index=codes["icd10_category"].values, columns=["response"])

        tokenizer = MistralTokenizer.from_hf_hub(model_id)
        model = Mistral3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        for _, row in tqdm(codes.iterrows(), total=len(codes)):
            query = TEMPLATE_MULTI.format(
                row["icd10_category"], 
                row["description"], 
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_MULTI},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query,
                        }
                    ],
                },
            ]

            for i in range(n_attempts):
                try:
                    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))

                    input_ids = torch.tensor([tokenized.tokens])
                    attention_mask = torch.ones_like(input_ids)

                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=8192,
                        temperature=0.15
                    )[0]

                    decoded_output = tokenizer.decode(output[len(tokenized.tokens):])
                    scores.loc[row["icd10_category"], "response"] = decoded_output
                    break
                    
                except Exception as e:
                    print("Attempt {} for code {}".format(i+1, row["icd10_category"]))
                    print(e)

            if not os.path.exists(log_dir): os.makedirs(log_dir)
            with open("{}/{}.txt".format(log_dir, row["icd10_category"]), "w") as f: print(json.dumps({"query": query, "response": decoded_output}, indent=4), file=f)
                
        return scores

    except Exception as e:
        print(e)
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
                       help="Model")
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

    OUTPUTS_DIR = "outputs"
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)
    
    for i in range(args.iterations):
        print(f"Getting responses on iteration {i}")
        responses = get_responses_multi(codes, args.model, log_dir=f"logs/logs_{args.model.replace('/', '_')}_{i}/")
        responses.to_csv(f"{OUTPUTS_DIR}/responses_{args.model.replace('/', '_')}_{i}.tsv", sep="\t")






# In this situation, you are playing a Pokémon game where your Pikachu (Level 42) is facing a wild Pidgey (Level 17). Here are the possible actions you can take and an analysis of each:

# 1. **FIGHT**:
#    - **Pros**: Pikachu is significantly higher level than the wild Pidgey, which suggests that it should be able to defeat Pidgey easily. This could be a good opportunity to gain experience points and possibly items or money.
#    - **Cons**: There is always a small risk of Pikachu fainting, especially if Pidgey has a powerful move or a status effect that could hinder Pikachu. However, given the large level difference, this risk is minimal.

# 2. **BAG**:
#    - **Pros**: You might have items in your bag that could help in this battle, such as Potions, Poké Balls, or Berries. Using an item could help you capture Pidgey or heal Pikachu if needed.
#    - **Cons**: Using items might not be necessary given the level difference. It could be more efficient to just fight and defeat Pidgey quickly.

# 3. **POKÉMON**:
#    - **Pros**: You might have another Pokémon in your party that is better suited for this battle or that you want to gain experience. Switching Pokémon could also be strategic if you want to train a lower-level Pokémon.
#    - **Cons**: Switching Pokémon might not be necessary since Pikachu is at a significant advantage. It could also waste time and potentially give Pidgey a turn to attack.

# 4. **RUN**:
#    - **Pros**: Running away could be a quick way to avoid the battle altogether. This might be useful if you are trying to conserve resources or if you are in a hurry to get to another location.
#    - **Cons**: Running away means you miss out on the experience points, items, or money that you could gain from defeating Pidgey. It also might not be the most efficient use of your time if you are trying to train your Pokémon.

# ### Recommendation:
# Given the significant level advantage, the best action to take is likely **FIGHT**. This will allow you to quickly defeat Pidgey and gain experience points for Pikachu. If you are concerned about Pikachu's health, you could use the **BAG** to heal Pikachu before or during the battle. Running away or switching Pokémon does not seem necessary in this situation.
