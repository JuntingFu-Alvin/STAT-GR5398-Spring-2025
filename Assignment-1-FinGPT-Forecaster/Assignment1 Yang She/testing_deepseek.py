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

deepseek_base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
)
deepseek_base_model.eval()

deepseek_model = PeftModel.from_pretrained(
    deepseek_base_model,
    "/root/FinGPT/finetuned_models/dow30-202305-202405-deepseek-8B_202502050341",
    cache_dir=cache_dir, 
    torch_dtype=torch.float16,
)
deepseek_model.eval()

deepseek_tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    cache_dir = cache_dir,
    truncation=True
)
deepseek_tokenizer.padding_side = "right"
deepseek_tokenizer.pad_token_id = deepseek_tokenizer.eos_token_id

test_dataset = load_dataset("dow30-202305-202405", from_remote=True)[0]["test"]
test_dataset = apply_to_all_prompts_in_dataset(test_dataset)

def test_acc(test_dataset, modelname):
    answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned = [], [], [], [], []
    if modelname == "deepseek":
        base_model = deepseek_base_model
        model = deepseek_model
        tokenizer = deepseek_tokenizer
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

deepseek_answers_base, deepseek_answers_fine_tuned, deepseek_gts, deepseek_times_base, deepseek_times_fine_tuned = test_acc(test_dataset, "deepseek")

print("\nEvaluating Base Model...")
deepseek_base_metrics = calc_metrics(deepseek_answers_base, deepseek_gts)

print("\nEvaluating Fine-Tuned Model...")
deepseek_fine_tuned_metrics = calc_metrics(deepseek_answers_fine_tuned, deepseek_gts)

print("\nBase Model Metrics:")
print(deepseek_base_metrics)

print("\nFine-Tuned Model Metrics:")
print(deepseek_fine_tuned_metrics)

with open("./results/deepseek_answers_base.pkl", "wb") as f:
    pickle.dump(deepseek_answers_base, f)

with open("./results/deepseek_answers_fine_tuned.pkl", "wb") as f:
    pickle.dump(deepseek_answers_fine_tuned, f)

with open("./results/deepseek_base_metrics.pkl", "wb") as f:
    pickle.dump(deepseek_base_metrics, f)

with open("./results/deepseek_fine_tuned_metrics.pkl", "wb") as f:
    pickle.dump(deepseek_fine_tuned_metrics, f)

with open("./results/deepseek_times_base.pkl", "wb") as f:
    pickle.dump(deepseek_times_base, f)

with open("./results/deepseek_times_fine_tuned.pkl", "wb") as f:
    pickle.dump(deepseek_times_fine_tuned, f)