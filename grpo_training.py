import argparse
import pandas as pd
import datasets
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.backends import cuda
from torch import bfloat16
import os
import random
from trl import GRPOConfig, GRPOTrainer

random.seed(100)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cuda.matmul.allow_tf32 = True


def save_samples(completions, ground_truth, log_file):
    for completion, gt in zip(completions, ground_truth):
        if random.random() < 0.1:  # 10% chance to save
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"\n\n==============\n")
                f.write(f"Prompt:\n{gt}\n\n")
                f.write(f"Completion:\n{completion}\n")
    return


def reward_length(completions, ground_truth, **kwargs):
    try:
        completions = [completion[0]["content"] for completion in completions]
        completions_length = [len(completion.split(' ')) for completion in completions]
        length_score = [
            1 - abs(400 - length) / 200 if 200 <= length <= 600
            else -abs(400 - length) / 50
            for length in completions_length
        ]
        log_file = kwargs.get("completion_log_file", "completion_samples/completion_samples.txt")
        save_samples(completions, ground_truth, log_file)
    except Exception as e:
        print("Error during reward scoring or sample saving:", e)
        length_score = [0] * len(completions)
    return length_score


def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['prompt'] = df['prompt'].apply(lambda x: [{"role": "user", "content": x}])
    return df


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    train_data = datasets.Dataset.from_pandas(prepare_data(args.data_path)).shuffle(seed=args.seed)

    timestamp = time.strftime("%Y%m%d_%H")
    modelname = re.search(r'(\d+B)', args.model_id).group(0) if re.search(r'(\d+B)', args.model_id) else "model"
    out_dir = f'{modelname}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=out_dir,
        deepspeed=args.deepspeed,
        overwrite_output_dir=args.overwrite_output_dir,
        seed=args.seed,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        tf32=args.tf32,
        bf16=args.bf16,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_on_each_node=args.save_on_each_node,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        optim=args.optim,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        num_generations=args.num_generations,
        remove_unused_columns=args.remove_unused_columns,
        log_completions=args.log_completions
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=bfloat16,
        low_cpu_mem_usage=True
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        reward_funcs=[lambda comps, gts, **kw: reward_length(comps, gts, completion_log_file=args.completion_log_file)]
    )

    try:
        trainer.train()
        trainer.save_model()
        print('Training DONE')
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, default="deepspeed_config.json")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_on_each_node", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--max_prompt_length", type=int, default=5000)
    parser.add_argument("--max_completion_length", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--remove_unused_columns", action="store_true")
    parser.add_argument("--log_completions", action="store_true")

    # New argument
    parser.add_argument("--completion_log_file", type=str, default="completion_samples/completion_samples.txt", help="Where to save sampled completions.")

    args = parser.parse_args()
    main(args)
