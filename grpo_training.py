import pandas as pd
import datasets
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import LoraConfig
from torch.backends import cuda
from torch import bfloat16
import os
import random
from trl import GRPOConfig, GRPOTrainer

random.seed(100)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cuda.matmul.allow_tf32 = True


def reward_length(completions,ground_truth, **kwargs):
    try: 
        completions = [completion[0]["content"] for completion in completions]
        completions_length=[len(completion.split(' ')) for completion in completions]
        length_score= [
            1 - abs(400 - length) / 200 if 200 <= length <= 600 
            else -abs(400 - length) / 50
            for length in completions_length 
        ]
    except:
        length_score= [0] * len(completions)
    
    # save_samples(completions,ground_truth)
    return length_score


def save_samples(completions,ground_truth):
    for completion,gt in zip(completions,ground_truth):
        if random.random() < 0.1:  # 1% chance to write samples into a file
            os.makedirs("completion_samples", exist_ok=True)
            log_file = os.path.join("completion_samples", "completion_samples.txt")
            with open(log_file, "a") as f:
                f.write(f"\n\n==============\n")
                f.write(f"{completion}")
    return


def prepare_data():
    """
    File with columns prompt and ground_truth
    """
    file='train_data.csv'

    df=pd.read_csv(file)
    ## convert prompt to conversational format
    df['prompt']=df['prompt'].apply(lambda x: [{"role": "user", "content": x}]) 
    return df
    


def main(model_id):
   
    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_data =datasets.Dataset.from_pandas(prepare_data()).shuffle(seed=42)

    print(train_data)  

    timestamp = time.strftime("%Y%m%d_%H")
    modelname=re.search(r'(\d+B)', model_id).group(0)
    out_dir = f'{modelname}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)


    params = {
    "output_dir": out_dir, 
    "deepspeed": "deepspeed_config.json",
    "overwrite_output_dir": True,
    "seed": 42,
    "logging_strategy": "steps",
    "logging_steps": 1,
    "learning_rate": 2e-5,
    "gradient_checkpointing": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "tf32": True,
    "bf16": True,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_on_each_node": False,
    "save_total_limit": 3,
    "report_to":"tensorboard",
    "push_to_hub": False,
    "optim": "adamw_torch",
    "max_prompt_length":5000,
    "max_completion_length":2000,
    "temperature":0.6,
    "num_generations":2,
    "remove_unused_columns":False,
    "log_completions":True,
    }   


    training_args = GRPOConfig(**params)

    model = AutoModelForCausalLM.from_pretrained(model_id,
        torch_dtype=bfloat16,
        low_cpu_mem_usage=True,
        )


    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        reward_funcs=[reward_length],  # can pass multiple functions
    )

    trainer.train()
    trainer.save_model()
    print('Training DONE')


if __name__ == "__main__":  
    model_id='Meta-Llama-3.1-8B-Instruct/'
    main(model_id)



