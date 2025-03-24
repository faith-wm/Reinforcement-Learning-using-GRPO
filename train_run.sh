#!/bin/bash

python train_with_grpo.py \
    --model_id Meta-Llama-3.1-8B-Instruct \
    --data_path train_data.csv \
    --deepspeed deepspeed_config.json \
    --overwrite_output_dir \
    --seed 42 \
    --logging_strategy steps \
    --logging_steps 1 \
    --learning_rate 2e-5 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --tf32 \
    --bf16 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \
    --report_to tensorboard \
    --optim adamw_torch \
    --max_prompt_length 5000 \
    --max_completion_length 2000 \
    --temperature 0.6 \
    --num_generations 2 \
    --remove_unused_columns \
    --log_completions \
    --completion_log_file completion_samples/completion_log.txt
