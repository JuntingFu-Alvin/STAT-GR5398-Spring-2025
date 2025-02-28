# -*- coding: utf-8 -*-
"""
Data set : FinGPT/fingpt-forecaster-dow30-202305-202405  

transformation from llama2 specific chat template 

to llama3 : meta-llama/Llama-3.1-8B-Instruct

and deepseek :deepseek-ai/DeepSeek-R1-Distill-Llama-8B

specified template and save it to local directory (./data )

by Simon Liao 2025/2/27
"""



import os
import re
import pandas as pd
from datasets import load_dataset
from datasets import Dataset


# Set the environment variables directly in the code (Not Recommended For sensitive keys)
os.environ["HF_TOKEN"] = ""  # Replace with your actual Hugging Face token

"""# load data set"""

###Llama3 dataset transfrom from llama2 chat template
def transform_prompt(text: str) -> str:
    """
    Transform a LLaMA instruct-style prompt into the
    <|begin_of_text|>...<|eot_id|> format.
    """
    pattern = r"\[INST\]\s*<<SYS>>(.*?)<</SYS>>(.*?)[/]INST\]\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        system_text = match.group(1).strip()
        user_text = match.group(2).strip()

        # Construct the new prompt format
        new_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_text}<|eot_id|>"
        )
        return new_prompt
    else:
        # Return original text if pattern doesn't match
        return text


## deepseek dataset transformation from llama3

def clean_prompt_for_DS(prompt: str)-> str:
    """
    Removes custom tags from a prompt string, leaving only the system_text and user_text.

    Args:
        prompt (str): The prompt string containing custom tags.

    Returns:
        str: The cleaned prompt without the tags.
    """
    prompt = prompt.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "")
    prompt = prompt.replace("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "")
    prompt = prompt.replace("[<|eot_id|>", "")
    return prompt.strip()

# load llama2 Dow30 dataset
ds = load_dataset("FinGPT/fingpt-forecaster-dow30-202305-202405")

ds.save_to_disk("./data/fingpt-forecaster-dow30-202305-202405")

dfs = [ds[split].to_pandas() for split in ds.keys()]
df_all = pd.concat(dfs, ignore_index=True)
df= df_all.copy()

df["prompt"] = df["prompt"].apply(transform_prompt)
#print(df["prompt"].iloc[0])

# save llama3 dataset to local
dataset_llama3 = Dataset.from_pandas(df)
# Split into train (80%) and test (20%)
dataset_llama3 = dataset_llama3.train_test_split(test_size=0.2, seed=42)
dataset_llama3.save_to_disk("./data/fingpt-forecaster-dow30-202305-202405-llama3-Instruct")



# clean prompt for deepseek
df["prompt"] = df["prompt"].apply(clean_prompt_for_DS)
#print(df["prompt"].iloc[0])


## deepseek dataset transformation from llama3
dataset_DSR1 = Dataset.from_pandas(df)
# Split into train (80%) and test (20%)
dataset_DSR1 = dataset_DSR1.train_test_split(test_size=0.2, seed=42)
dataset_DSR1.save_to_disk("./data/fingpt-forecaster-dow30-202305-202405_Deepseek_r1_8B")