export NCCL_IGNORE_DISABLED_P2P=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0


deepspeed \
--include localhost:0 \
train_lora.py \
--run_name dow30v3-deepseek-5e-5lr-qkvogud \
--base_model deepseekR1 \
--dataset fingpt-forecaster-dow30-202305-202405-llama3-Instruct \
--max_length 4096 \
--batch_size 1 \
--gradient_accumulation_steps 16 \
--learning_rate 5e-5 \
--num_epochs 5 \
--log_interval 10 \
--warmup_ratio 0.03 \
--scheduler constant \
--evaluation_strategy steps \
--ds_config config.json