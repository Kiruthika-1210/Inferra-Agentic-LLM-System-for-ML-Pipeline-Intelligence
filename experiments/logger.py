import json
import os

def save_log(dataset_name, profile, insights, iterations, final_result):

    log = {
        "dataset": dataset_name,
        "dataset_profile": profile,
        "dataset_insights": insights,
        "iterations": iterations,
        "final_result": final_result
    }

    os.makedirs("experiments/logs", exist_ok=True)

    with open(f"experiments/logs/{dataset_name}_experiment_log.json", "w") as f:
        json.dump(log, f, indent=2)