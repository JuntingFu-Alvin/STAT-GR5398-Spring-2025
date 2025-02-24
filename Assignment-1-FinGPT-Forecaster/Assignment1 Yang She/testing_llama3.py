import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
from datasets import Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# from peft import PeftModel
from utils import *
import time
import pickle
from test_utils import *

os.environ["HUGGINGFACE_TOKEN"] = ""
hf_token = ""
cache_dir = "./pretrained-models"

llama3_base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
)
llama3_base_model.eval()

llama3_model = PeftModel.from_pretrained(
    llama3_base_model,
    "/root/FinGPT/finetuned_models/dow30-202305-202405-llama3.1-8B_202502020308",
    cache_dir=cache_dir, 
    torch_dtype=torch.float16,
)
llama3_model.eval()

llama3_tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    cache_dir = cache_dir,
)
llama3_tokenizer.padding_side = "right"
llama3_tokenizer.pad_token_id = llama3_tokenizer.eos_token_id

test_dataset = load_dataset("dow30-202305-202405", from_remote=True)[0]["test"]
test_dataset = apply_to_all_prompts_in_dataset(test_dataset)

def test_acc(test_dataset, modelname):
    answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned = [], [], [], [], []
    if modelname == "llama3":
        base_model = llama3_base_model
        model = llama3_model
        tokenizer = llama3_tokenizer
    else:
        return None

    for i in tqdm(range(len(test_dataset)), desc="Processing test samples"):
        try:
            prompt = test_dataset[i]['prompt']
            gt = test_dataset[i]['answer']

            output_base, time_base = test_demo(base_model, tokenizer, prompt)
            answer_base = re.sub(r'.*\[/INST\]\s*', '', output_base, flags=re.DOTALL)

            output_fine_tuned, time_fine_tuned = test_demo(model, tokenizer, prompt)
            answer_fine_tuned = re.sub(r'.*\[/INST\]\s*', '', output_fine_tuned, flags=re.DOTALL)

            answers_base.append(answer_base)
            answers_fine_tuned.append(answer_fine_tuned)
            gts.append(gt)
            times_base.append(time_base)
            times_fine_tuned.append(time_fine_tuned)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    return answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned

llama3_answers_base, llama3_answers_fine_tuned, llama3_gts, llama3_times_base, llama3_times_fine_tuned = test_acc(test_dataset, "llama3")

print("\nEvaluating Base Model...")
llama3_base_metrics = calc_metrics(llama3_answers_base, llama3_gts)

print("\nEvaluating Fine-Tuned Model...")
llama3_fine_tuned_metrics = calc_metrics(llama3_answers_fine_tuned, llama3_gts)

print("\nBase Model Metrics:")
print(llama3_base_metrics)

print("\nFine-Tuned Model Metrics:")
print(llama3_fine_tuned_metrics)

with open("./results/llama3_answers_base.pkl", "wb") as f:
    pickle.dump(llama3_answers_base, f)

with open("./results/llama3_answers_fine_tuned.pkl", "wb") as f:
    pickle.dump(llama3_answers_fine_tuned, f)

with open("./results/llama3_base_metrics.pkl", "wb") as f:
    pickle.dump(llama3_base_metrics, f)

with open("./results/llama3_fine_tuned_metrics.pkl", "wb") as f:
    pickle.dump(llama3_fine_tuned_metrics, f)

with open("./results/llama3_times_base.pkl", "wb") as f:
    pickle.dump(llama3_times_base, f)

with open("./results/llama3_times_fine_tuned.pkl", "wb") as f:
    pickle.dump(llama3_times_fine_tuned, f)