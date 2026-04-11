import os
import pandas as pd
import subprocess
from urllib.parse import urlparse
from experiments.baselines import run_manual_baseline, run_gridsearch_baseline


def run_all(file_path, target):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ----------------------------
    # Dataset name extraction
    # ----------------------------
    if file_path.startswith("http"):
        dataset_name = os.path.splitext(
            os.path.basename(urlparse(file_path).path)
        )[0]
    else:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    # ----------------------------
    # Create dataset-specific folder
    # ----------------------------
    dataset_dir = os.path.join(BASE_DIR, "experiments", "results", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # ----------------------------
    # Paths (per dataset)
    # ----------------------------
    manual_path = os.path.join(dataset_dir, "manual.csv")
    grid_path = os.path.join(dataset_dir, "gridsearch.csv")
    agentic_path = os.path.join(dataset_dir, "agentic.csv")
    summary_path = os.path.join(dataset_dir, "summary.csv")

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(file_path)

    # ----------------------------
    # Run baselines
    # ----------------------------
    run_manual_baseline(df, target, dataset_name, manual_path)
    run_gridsearch_baseline(df, target, dataset_name, grid_path)

    # ----------------------------
    # Run agentic system
    # ----------------------------
    subprocess.run([
        "python", "main.py",
        "--file", file_path,
        "--target", target
    ], check=True)

    # ----------------------------
    # Load results
    # ----------------------------
    manual = pd.read_csv(manual_path)
    grid = pd.read_csv(grid_path)

    # ⚠️ Ensure main.py saves here
    if not os.path.exists(agentic_path):
        raise FileNotFoundError(
            f"Agentic results not found at {agentic_path}. "
            "Make sure main.py saves results correctly."
        )

    agentic = pd.read_csv(agentic_path)

    # ----------------------------
    # Create summary
    # ----------------------------
    summary = pd.DataFrame([
        {
            "dataset": dataset_name,
            "method": "manual",
            "accuracy": manual["test_accuracy"][0],
            "f1": manual["f1_score"][0],
            "runtime": manual["runtime"][0]
        },
        {
            "dataset": dataset_name,
            "method": "gridsearch",
            "accuracy": grid["test_accuracy"][0],
            "f1": grid["f1_score"][0],
            "runtime": grid["runtime"][0]
        },
        {
            "dataset": dataset_name,
            "method": "agentic",
            "accuracy": agentic["test_accuracy"].max(),
            "f1": agentic["f1_score"].max(),
            "runtime": agentic["runtime"].mean()
        }
    ])

    summary.to_csv(summary_path, index=False)

    print(f"\nExperiment Completed for {dataset_name}")
    print(f"Results saved in: {dataset_dir}")
    print("━"*50)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--target", required=True)

    args = parser.parse_args()

    run_all(args.file, args.target)