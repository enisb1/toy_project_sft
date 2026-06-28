from load_grpo_data import load_data
import re
import torch
from dotenv import load_dotenv
from peft import LoraConfig, AutoPeftModelForCausalLM
load_dotenv()

from vllm import LLM, SamplingParams

import mlflow
from trl import GRPOConfig, GRPOTrainer

import os


OUTPUT_DIR = "outputs/grpo_training"
MERGED_DIR = "outputs/grpo_merged"
MODEL_ID = "ibm-granite/granite-4.0-350M"

RUN_NAME = "granite-350m-grpo-csqa"
MLFLOW_TRACKING_URI = "https://mlflow.vramfullagain.lol"
MLFLOW_EXPERIMENT = "csqa-granite-grpo"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

TRAIN_SUBSET = 100  # how many samples to gather from dataset

LETTER_PATTERN = re.compile(r"Output letter:\s*([A-E])\b")
PLAINTEXT_PATTERN = re.compile(r"Output text:\s*(.+)")

letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def correct_text(choices, answer_key):
    """Return the plain-text of the gold option for one example."""
    texts = choices["text"]  # the five option strings
    return texts[letter_to_index[answer_key]]


def norm(s):
    return s.strip().lower()


# --------------------------------------------------------------------------- #
# 3. The two reward functions (0.5 each)
# --------------------------------------------------------------------------- #
def letter_reward(completions, answerKey, **kwargs):
    """
    Completion contains 'Output text: <TEXT>' AND that text correct one
    """
    rewards = []
    for completion, ans_key in zip(completions, answerKey):
        m = LETTER_PATTERN.search(completion)
        if m and m.group(1).upper() == ans_key.upper():
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def plaintext_reward(completions, choices, answerKey, **kwargs):
    """
    Completion contains 'Output text: <TEXT>' AND that text correct one
    """
    rewards = []
    for completion, choice_set, ans_key in zip(completions, choices, answerKey):
        m = PLAINTEXT_PATTERN.search(completion)
        gold = correct_text(choice_set, ans_key)
        if m and gold is not None and norm(m.group(1)) == norm(gold):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def main():
    load_dotenv()
        
    ds = load_data(TRAIN_SUBSET)

    training_args = GRPOConfig(
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_generations=8,  # completions sampled per prompt
        max_completion_length=128,
        reward_weights=[0.6, 0.4],
        report_to="mlflow",
        run_name=RUN_NAME,
        log_completions=True,
        num_completions_to_print=3
    )
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=[letter_reward, plaintext_reward],
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config
    )

    trainer.train()
    
    # Merge LoRA into base weights and save in MERGED_DIR (can be used for inference)
    print("Merging adapter into base model ->", MERGED_DIR)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR)
    trainer.processing_class.save_pretrained(MERGED_DIR)


if __name__ == "__main__":
    main()
