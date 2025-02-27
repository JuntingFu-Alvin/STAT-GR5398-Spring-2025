from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainerState, TrainerControl
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import os
import re
import sys
import wandb
import argparse
from datetime import datetime
from functools import partial
from tqdm import tqdm
from utils import *

# LoRA
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,  
    set_peft_model_state_dict,   
)

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable HuggingFace progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Replace with your own API key and project name
os.environ['WANDB_API_KEY'] = ''  
os.environ['WANDB_PROJECT'] = 'fingpt-forecaster-lora-fine-tuning_3'


class GenerationEvalCallback(TrainerCallback):
    
    def __init__(self, eval_dataset, tokenizer, ignore_until_epoch=0):
        self.eval_dataset = eval_dataset
        self.ignore_until_epoch = ignore_until_epoch
        self.tokenizer = tokenizer
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"\n=== Evaluation at step {state.global_step}, epoch {state.epoch} ===")
        
        if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
            print(f"Skipping evaluation (waiting until epoch {self.ignore_until_epoch})")
            return
                
        if state.is_local_process_zero:
            try:
                model = kwargs['model']
                tokenizer = self.tokenizer
                generated_texts, reference_texts = [], []
                parsing_success_count = 0
                parsing_failure_count = 0

                print(f"Generating predictions for {len(self.eval_dataset)} examples...")
                for i, feature in enumerate(tqdm(self.eval_dataset)):
                    try:
                        prompt = feature['prompt']
                        gt = feature['answer']
                        inputs = tokenizer(
                            prompt, return_tensors='pt',
                            padding=False, max_length=4096
                        )
                        inputs = {key: value.to(model.device) for key, value in inputs.items()}
                        
                        res = model.generate(
                            **inputs, 
                            use_cache=True,
                            max_new_tokens=512  # Add a max token limit
                        )
                        output = tokenizer.decode(res[0], skip_special_tokens=True)
                        answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

                        # Verify parsing for debugging (just try it without storing result)
                        gen_parsed = parse_answer(answer)
                        ref_parsed = parse_answer(gt)
                        
                        if gen_parsed and ref_parsed:
                            parsing_success_count += 1
                        else:
                            parsing_failure_count += 1
                            if i < 3:  # Only print first few failures to avoid spam
                                print(f"\n--- Example {i}: Parsing {'failed' if not gen_parsed else 'succeeded'} for generated text, {'failed' if not ref_parsed else 'succeeded'} for reference ---")
                                print(f"GENERATED (first 200 chars): {answer[:200]}...")
                                print(f"REFERENCE (first 200 chars): {gt[:200]}...")

                        generated_texts.append(answer)
                        reference_texts.append(gt)
                        
                    except Exception as e:
                        print(f"Error generating prediction for example {i}: {e}")
                
                print(f"\nParsing results: {parsing_success_count} successes, {parsing_failure_count} failures")
                
                try:
                    print(f"Calculating metrics...")
                    metrics = calc_metrics(generated_texts, reference_texts)
                    print(f"Raw metrics: {metrics}")
                    
                    if not metrics:
                        print("WARNING: No metrics returned. This usually means all parsing failed.")
                        # Log a placeholder so something appears
                        wandb.log({"custom/parsing_success_rate": parsing_success_count / len(self.eval_dataset)}, step=state.global_step)
                        return
                    
                    # Create better namespaced metrics
                    flattened_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                flattened_metrics[f"custom/{key}/{nested_key}"] = nested_value
                        else:
                            flattened_metrics[f"custom/{key}"] = value
                    
                    print(f"Logging metrics to W&B: {flattened_metrics}")
                    
                    # Ensure wandb is initialized
                    if wandb.run is None:
                        wandb.init(project=os.environ['WANDB_PROJECT'])
                        
                    wandb.log(flattened_metrics, step=state.global_step)
                    
                except Exception as e:
                    print(f"Error in metrics calculation: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"Error in evaluation: {e}")
                import traceback
                traceback.print_exc()
                
            torch.cuda.empty_cache()


def main(args):
        
    model_name = parse_model_name(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load Dataset from Local Storage
    dataset_path = args.dataset
    dataset = datasets.load_from_disk(dataset_path)

    dataset_train = dataset["train"].shuffle(seed=42)
    dataset_test = dataset["test"]

    original_dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    
    eval_dataset = original_dataset['test'].shuffle(seed=42).select(range(50))
    
    dataset = original_dataset.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', len(dataset['train']))
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(
        ['prompt', 'answer', 'label', 'symbol', 'period', 'exceed_max_length']
    )
    
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')
    
    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}',
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
        run_name=args.run_name
    )
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.model.config.use_cache = False

    # Setup PEFT
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
                tokenizer=tokenizer,  # Pass tokenizer here
                ignore_until_epoch=round(0.3 * args.num_epochs)
            )
        ]
    )
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    torch.cuda.empty_cache()
    trainer.train()

    # Save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2', 'llama3.1', 'deepseek'])  
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="The train batch size per device")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--num_epochs", default=8, type=float, help="The training epochs")
    parser.add_argument("--num_workers", default=8, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--ds_config", default=os.path.abspath('./config.json'), type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--evaluation_strategy", default='steps', type=str)    
    parser.add_argument("--eval_steps", default=0.1, type=float)   

    args = parser.parse_args()
    
    wandb.login()
    main(args)
