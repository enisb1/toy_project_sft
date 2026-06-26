from load_grpo_data import load_data
import re
import mlflow
from trl import GRPOConfig, GRPOTrainer

OUTPUT_DIR = "outputs/grpo_training"
MODEL_ID = "ibm-granite/granite-4.0-350M"

RUN_NAME = "granite-350m-grpo-csqa"
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "csqa-granite-grpo"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

TRAIN_SUBSET = 10  # how many samples to gather from dataset

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
    ds = load_data(TRAIN_SUBSET)
    print(ds)
    print(set(ds["answerKey"]))

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_generations=8,  # completions sampled per prompt
        max_completion_length=128,
        logging_steps=10,
        reward_weights=[0.6, 0.4],
        report_to="mlflow",
        run_name=RUN_NAME,
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=[letter_reward, plaintext_reward],
        args=training_args,
        train_dataset=ds,
    )

    trainer.train()


if __name__ == "__main__":
    main()
