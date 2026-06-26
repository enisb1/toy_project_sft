from datasets import load_dataset

SYSTEM_INSTRUCTIONS = (
    "Answer the multiple-choice question. Use exactly this format:\n"
    "Output letter: <LETTER>\n"
    "Output text: <the answer text>"
)


def format_example(example):
    labels = example["choices"]["label"]  # ["A","B","C","D","E"]
    texts = example["choices"]["text"]
    options = "\n".join(f"{lab}: {txt}" for lab, txt in zip(labels, texts))
    prompt = f"{SYSTEM_INSTRUCTIONS}\n\nQuestion: {example['question']}\n\nOptions:\n{options}"
    return {"prompt": prompt}


def load_data(TRAIN_SUBSET):
    ds = load_dataset("tau/commonsense_qa", split="train")

    ds = ds.remove_columns(["id", "question_concept"])

    if TRAIN_SUBSET is not None:
        ds = ds.select(range(TRAIN_SUBSET))

    ds = ds.map(format_example)

    return ds
