import datasets
import re
import time
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.backends import cuda
from torch import bfloat16, float16
import os
import deepspeed
from torch.utils.data import DataLoader
import argparse
# from mpi4py import MPI
import torch
from torch.utils.data import DataLoader, RandomSampler
from trl import SFTConfig, SFTTrainer
import random
from trl import GRPOConfig, GRPOTrainer, OnlineDPOConfig, OnlineDPOTrainer

# 
random.seed(100)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cuda.matmul.allow_tf32 = True



# import wandb

from dataprep import hpo_data, reward_length, reward_hpo_distance

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


def main(name):
    model_id='Llama3/Meta-Llama-3.1-8B-Instruct/'
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    train_data = datasets.Dataset.from_pandas(hpo_data(tokenizer)).shuffle(seed=42)
    print(train_data)

    timestamp = time.strftime("%Y%m%d_%H")
    filename=re.search(r'(\d+B)', model_id).group(0)
    out_dir = f'finetuned_models/{filename}_{timestamp}_{name}'
    os.makedirs(out_dir, exist_ok=True)

    ######
    # wandb.init(project=f"{filename}_{timestamp}_{name}")
    ######

    params = {
    # "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "output_dir": out_dir, 
    "deepspeed": "single_node_8b.json",
    "overwrite_output_dir": True,
    "seed": 42,
    "logging_strategy": "steps",
    "logging_steps": 1,
    "learning_rate": 5e-7,
    "gradient_checkpointing": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "tf32": True,
    "bf16": True,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_on_each_node": False,
    "save_total_limit": 3,
    "report_to": "tensorboard",#["tensorboard", "wandb"],
    "push_to_hub": False,
    # "optim": "adamw_torch",
    # "max_prompt_length":5000,
    "max_completion_length":2000,
    "temperature":0.6,
    # "num_generations":4,
    "remove_unused_columns":False,
    "log_completions":True,
    ####
    # "missing_eos_penalty":1,
    # "max_new_tokens":1000
    ######
    #  "use_vllm":True,
    # "vllm_device":"cuda:7",
    # "vllm_gpu_memory_utilization":0.3,
    #  "num_generations":7,
        # "max_length":20000,
    # "vllm_max_model_len":20000,
        # "warmup_ratio": 0.2, 
          # "weight_decay": 0.05,' 
          # # "num_train_epochs": 20, 
              # "lr_scheduler_type": "cosine", 
            # "max_grad_norm": 0.3,
    # "do_eval":False,
    }   


    training_args = GRPOConfig(**params)

    # params['rank']=128
    # params['alpha']=256
    params['setting']='GRPO'
    params['train']=train_data
    # params['test']=test_data

    param_file_path = os.path.join(out_dir, "parameters.txt")
    with open(param_file_path,'w') as parameters_file:
        for k,v in params.items():
            parameters_file.write(f'{k}:{v}\n')
    print('Parameters written to file')

    
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
    
    # peft_config = LoraConfig(
    #     r=params['rank'],
    #     lora_alpha=params['alpha'],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=['down_proj', 'gate_proj', 'o_proj', 'v_proj', 'up_proj', 'q_proj', 'k_proj']
    #     )
    

    model = AutoModelForCausalLM.from_pretrained(model_id,
        torch_dtype=bfloat16,
        low_cpu_mem_usage=True,
        # quantization_config=bnb_config,
        )
    

    # model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)
    # model.config.use_cache = False
    # model = get_peft_model(model, peft_config)
    # # print(model.print_trainable_parameters())

    # model.config.use_cache = False
    # model.config.pretraining_tp = 1


    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        reward_funcs=[reward_hpo_distance],
        processing_class=tokenizer,
        # peft_config=peft_config
    )

    trainer.train()
    trainer.save_model()
    print('Training DONE')


main(name=f'llama_norm_only')



