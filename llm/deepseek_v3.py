import pandas as pd
from tqdm import tqdm
import json
from openai import OpenAI

api_url = "https://api.modelarts-maas.com/v1/chat/completions"
model = "DeepSeek-V3"
with open("api_key.txt", "r") as f: api_key = f.readline().strip()

base_url = "https://api.modelarts-maas.com/v1" # API URL
api_key = api_key # Replace yourApiKey with the obtained API Key
client = OpenAI(api_key=api_key, base_url=base_url)

SYSTEM_PROMPT = """I'll give you pairs of ICD-10 codes and thier descriptions. You have to tell me, if a patient has one of them in his medical history, how likely is it that there will be another. 
ANSWER IN JSON FORMAT:
{
    "comment": <your thoughts and explanations>,
    "answer": <low/medium/high>
}
DO NOT ADD ANYTHING IN YOUR ANSWER."""

TEMPLATE = """{{
    icd_code_a: {},
    icd_code_a_description: {},
    icd_code_b: {},
    icd_code_b_description: {}
}}"""

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

def get_scores(codes):
    try:
        scores = pd.DataFrame(data=[["null" for _ in range(len(codes))] for _ in range(len(codes))], index=codes["icd10_category"].values, columns=codes["icd10_category"].values)

        for i, row_i in tqdm(codes.iterrows()):
            for j, row_j in codes.iloc[i+1:].iterrows():
                query = TEMPLATE.format(
                    row_i["icd10_category"], 
                    row_i["description"], 
                    row_j["icd10_category"], 
                    row_j["description"]
                )

                response = client.chat.completions.create(
                    model = "DeepSeek-V3", # model Parameter
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    temperature = 0.3,
                    stream = False
                )
                
                scores.loc[row_i["icd10_category"], row_j["icd10_category"]] = response.choices[0].message.content
                
                with open("logs/{}_{}.txt".format(row_i["icd10_category"], row_j["icd10_category"]), "w") as f: json.dump({"query": query, "response": response.choices[0].message.content}, f)
                
        return scores

    except Exception as e:
        print(e)
        return scores
    
def get_responses_multi(codes):
    try:
        scores = pd.DataFrame(data=["null" for _ in range(len(codes))], index=codes["icd10_category"].values, columns=["response"])

        for _, row in tqdm(codes.iterrows()):
            query = TEMPLATE_MULTI.format(
                row["icd10_category"], 
                row["description"], 
            )

            response = client.chat.completions.create(
                model = "DeepSeek-V3", # model Parameter
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_MULTI},
                    {"role": "user", "content": query},
                ],
                temperature = 0.3,
                stream = False
            )

            scores.loc[row["icd10_category"], "response"] = response.choices[0].message.content

            with open("logs/{}.txt".format(row["icd10_category"]), "w") as f: json.dump({"query": query, "response": response.choices[0].message.content}, f)
                
        return scores

    except Exception as e:
        print(e)
        return scores
    
codes = pd.read_csv("icd10_categories_descriptions.csv").drop("Unnamed: 0", axis=1)

responses = get_responses_multi(codes)
responses.to_csv("responses.tsv", sep="\t")

# scores = get_scores(codes[:10])
# scores.to_csv("scores.tsv", sep="\t")

print("SUCCESS")