from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer import TRAINING_ARGS_NAME
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import os
import re
import sys
import wandb
import argparse
import pandas as pd
from datetime import datetime
from functools import partial
from tqdm import tqdm
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# LoRA
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,   
)

# Replace with your own api_key and project name
os.environ['WANDB_API_KEY'] = ''    # TODO: Replace with your environment variable
os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'
cache_dir = "./pretrained-models"

class GenerationEvalCallback(TrainerCallback):
    
    def __init__ (self, eval_dataset,tokenizer, ignore_until_epoch=0):
        self.eval_dataset = eval_dataset

        self.tokenizer=tokenizer
        self.ignore_until_epoch = ignore_until_epoch
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print('!!!!!!!!!!!!!!evaluating!!!!!!!')
        if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
            return

        if state.is_local_process_zero:
            model = kwargs['model']
            tokenizer = kwargs['processing_class']
            generated_texts, reference_texts = [], []

            '''with torch.no_grad():
                  output_tokens = model(**inputs)
                  logits = output_tokens.logits  # 提取 logits
                  print("Logits Mean:", logits.mean().item())
                  print("Logits Std:", logits.std().item())
                  print("Logits Max:", logits.max().item())
                  print("Logits Min:", logits.min().item())
            
            for name, param in model.named_parameters():
                  if 'proj' in name:
                    print(f"{name}: mean={param.mean().item()}, std={param.std().item()}")
            '''

            for feature in tqdm(self.eval_dataset):
                prompt = feature['prompt']
                gt = feature['answer']
                inputs = tokenizer(
                    prompt, return_tensors='pt',
                    padding=False, max_length=4096,
                    #truncation=True
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}

                
                
                
                res = model.generate(
                    **inputs, 
                    #max_new_tokens=200,
                    #temperature=0.6,
                    #eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,    
                )
                
  
                output = tokenizer.decode(res[0], skip_special_tokens=True)

                
                print('output:',output)
                answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

                generated_texts.append(answer)
                reference_texts.append(gt)

                print("GENERATED: ", answer)
                #print("REFERENCE: ", gt)
            print('before metrics')
            metrics = calc_metrics(reference_texts, generated_texts)
            print('after metrics')
            # Ensure wandb is initialized
            if wandb.run is None:
                wandb.init()
                
            wandb.log(metrics, step=state.global_step)
            torch.cuda.empty_cache()            


def main(args):
        
    #model_name = parse_model_name(args.base_model, args.from_remote)
    model_name = "meta-llama/Llama-3.1-8B"
    #model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_8bit=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
        device_map=None,
        torch_dtype=torch.bfloat16
    )
    if args.local_rank == 0:
        print(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    # load data
    if not args.from_remote:
      dataset_fname = "./data/fingpt-forecaster-" + args.dataset
    else:
      dataset_fname = args.dataset
    #dataset_list = load_dataset(dataset_fname, args.from_remote)
    dataset_list = load_dataset("FinGPT/fingpt-forecaster-dow30-202305-202405", args.from_remote)
    dataset_train = datasets.concatenate_datasets([d['train'] for d in dataset_list]).shuffle(seed=42)
    
    if args.test_dataset:
        test_dataset_fname = "./data/" + args.test_dataset
        dataset_list = load_dataset(test_dataset_fname, args.from_remote)
            
    dataset_test = datasets.concatenate_datasets([d['test'] for d in dataset_list])
    
    original_dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    
    eval_dataset = original_dataset['test'].shuffle(seed=42).select(range(50))
    
    dataset = original_dataset.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', len(dataset['train']))
    
    df = pd.DataFrame(dataset["train"])
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(
        ['prompt', 'answer','label', 'symbol', 'period', 'exceed_max_length']
    )
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')
    
    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}', # 保存位置
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        fp16=True,
        deepspeed=args.ds_config,
        evaluation_strategy=args.evaluation_strategy,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name,
    )
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.model.config.use_cache = False

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=lora_module_dict[args.base_model],
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    #print(dataset['train'])
    #print(dataset['test'])
    # Train

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'], 
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True,
            return_tensors="pt"
        ),
        callbacks=[
            GenerationEvalCallback(
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                ignore_until_epoch=round(0.2 * args.num_epochs)
            )
        ]
    )
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    torch.cuda.empty_cache()
    trainer.train()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2','llama31','dsllama'])
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="The train batch size per device")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--num_epochs", default=4, type=float, help="The training epochs")
    parser.add_argument("--num_workers", default=6, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--warmup_ratio", default=0.03, type=float)
    parser.add_argument("--ds_config", default='./config.json', type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)    
    parser.add_argument("--eval_steps", default=0.1, type=float)    
    parser.add_argument("--from_remote", default=False, type=bool)    
    args = parser.parse_args()
    
    wandb.login()
    main(args)