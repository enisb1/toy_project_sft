import torch
import mlflow
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from dataloader import load_data
from patched_tokenizer import get_patched_tokenizer

MODEL_ID = "ibm-granite/granite-4.0-350M"
OUTPUT_DIR = "outputs/sft-granite"
MERGED_DIR = OUTPUT_DIR + "-merged"  # full model for vLLM to serve

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "ultrafeedback-granite-sft"
RUN_NAME = "granite-lora-sft-subset-400"

# keeping toy task small. can bump these up once the pipeline runs end to end.
TRAIN_SUBSET = 5000  # set to None to use all 61k rows
TEST_SUBSET = 100
NUM_EPOCHS = 1
MAX_LENGTH = 1024


def train():
    print("Loading data")
    ds = load_data(TRAIN_SUBSET, TEST_SUBSET)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",  # let peft pick the linear layers
    )

    # to use only assistant's loss, you
    patched_tok = get_patched_tokenizer()

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 16
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_length=MAX_LENGTH,
        assistant_only_loss=True,  # loss on assistant turns only
        packing=False,  # keep off; packing + masking is fiddly
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        eval_strategy="no",
        eval_steps=5,
        save_strategy="epoch",
        report_to="mlflow",  # built-in callback auto-logs args+metrics
        run_name=RUN_NAME,
        model_init_kwargs={"dtype": torch.bfloat16},
    )

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=peft_config,
        processing_class=patched_tok,
    )

    print("Starting training")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME):
        # Things the trainer's callback won't capture -> log them ourselves.
        mlflow.log_params(
            {
                "model_id": MODEL_ID,
                "dataset": "HuggingFaceH4/ultrafeedback_binarized:train_sft",
                "train_examples": len(ds["train"]),
                "eval_examples": len(ds["test"]),
                "lora_r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "lora_target": peft_config.target_modules,
                "assistant_only_loss": args.assistant_only_loss,
            }
        )

        trainer.train()  # auto-logs metrics + training args into this run

        # Save the adapter.
        trainer.save_model(OUTPUT_DIR)

        # Merge LoRA into base weights so vLLM can serve a plain model dir.
        print("Merging adapter into base model ->", MERGED_DIR)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(MERGED_DIR)
        trainer.processing_class.save_pretrained(MERGED_DIR)

        # Log the merged model directory as an artifact so the run is
        # self-contained (config + metrics + the actual model).
        mlflow.log_artifacts(MERGED_DIR, artifact_path="merged_model")

        # run_id needed in eval so to save metrics for this model
        run_id = mlflow.active_run().info.run_id
        print("=" * 60)
        print("MLflow run_id (paste into evaluate.py):", run_id)
        print("=" * 60)

        print("Done. Serve this dir with vLLM:", MERGED_DIR)


if __name__ == "__main__":
    train()
