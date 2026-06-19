from datasets import load_dataset


def load_data(TRAIN_SUBSET, TEST_SUBSET):
    ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split={"train": "train_sft", "test": "test_sft"},
    )

    # remove not needed columns
    ds = ds.remove_columns([c for c in ds["train"].column_names if c != "messages"])

    if TRAIN_SUBSET is not None:
        ds["train"] = ds["train"].select(range(TRAIN_SUBSET))
        ds["test"] = ds["test"].select(range(TEST_SUBSET))

    return ds


if __name__ == "__main__":
    load_data()
