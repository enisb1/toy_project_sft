"""
Usage:
  1) run train.py, copy the printed run_id
  2) paste it into TRAINING_RUN_ID below
  3) python src/evaluate.py
"""

import json
import mlflow
from lm_eval import simple_evaluate

# ---- paste the run_id printed at the end of train.py ----
TRAINING_RUN_ID = "4f188269f2b448e28391276c6fb7f020"

MLFLOW_TRACKING_URI = "http://localhost:5000"  # same server as training
BASE_MODEL = "ibm-granite/granite-4.0-350M"
SFT_MODEL = "outputs/sft-granite-merged"
TASKS = ["arc_easy", "mmlu_abstract_algebra"]
LIMIT = 200  # examples per task; keep small for a fast shakedown


def run_eval(model_path: str) -> dict:
    """Run lm-eval on one model, return {task: {metric: value}}."""
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=TASKS,
        limit=LIMIT,
        batch_size=8,
        device="cuda:0",
    )
    return results["results"]


def extract_acc(task_results: dict) -> dict:
    """Pull the primary accuracy metric for each task."""
    out = {}
    for task, metrics in task_results.items():
        # lm-eval metric keys look like "acc,none" / "acc_norm,none"
        for key in ("acc,none", "acc_norm,none"):
            if key in metrics:
                out[task] = metrics[key]
                break
    return out


def main():
    print("Evaluating BASE model:", BASE_MODEL)
    base = extract_acc(run_eval(BASE_MODEL))
    print("Base:", base)

    print("\nEvaluating SFT model:", SFT_MODEL)
    sft = extract_acc(run_eval(SFT_MODEL))
    print("SFT:", sft)

    # Log both, plus the delta, into the SAME training run.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=TRAINING_RUN_ID):
        for task in TASKS:
            b, s = base.get(task), sft.get(task)
            mlflow.log_metric(f"eval_base_{task}", b)
            mlflow.log_metric(f"eval_sft_{task}", s)
            mlflow.log_metric(f"eval_delta_{task}", s - b)
            print(f"{task}: base={b:.4f}  sft={s:.4f}  delta={s - b:+.4f}")

        # Also save the raw scores as an artifact.
        with open("eval_results.json", "w") as f:
            json.dump({"base": base, "sft": sft}, f, indent=2)
        mlflow.log_artifact("eval_results.json")

    print("\nLogged eval results into MLflow run:", TRAINING_RUN_ID)


if __name__ == "__main__":
    main()
