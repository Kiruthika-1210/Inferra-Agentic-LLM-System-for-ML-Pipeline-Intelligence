import pandas as pd
import json
import os

from utils.llm_interface import call_llm
from agents.judge_agent import judge_profile

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def compute_basic_stats(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute dataset statistics
    """

    num_samples = X.shape[0]
    num_features = X.shape[1]

    missing_percentage = (
        X.isnull().sum().sum() / (num_samples * num_features)
    ) * 100

    categorical_features = X.select_dtypes(include=["object"]).shape[1]
    numerical_features = X.select_dtypes(exclude=["object"]).shape[1]

    class_distribution = y.value_counts(normalize=True).to_dict()

    stats = {
        "samples": num_samples,
        "features": num_features,
        "missing_percentage": round(missing_percentage, 2),
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "class_distribution": class_distribution
    }

    return stats


def build_prompt(stats: dict) -> str:
    return f"""
You are a data scientist.

Given this dataset summary:
{json.dumps(stats, indent=2)}

Analyze and return:

1. dataset_size: small / medium / large
2. imbalance: low / moderate / high
3. missing_severity: low / moderate / high
4. feature_complexity: numerical / categorical / mixed

Return ONLY valid JSON in this format:
{{
  "dataset_size": "...",
  "imbalance": "...",
  "missing_severity": "...",
  "feature_complexity": "..."
}}

All fields are mandatory.
Do not omit any field.
Do not include explanations.
Ensure JSON is strictly valid.
"""

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def profile_data(X: pd.DataFrame, y: pd.Series, dataset_name) -> dict:

    cached = load_json(f"results/profile/{dataset_name}_profile.json")
    if cached:
        print("\nUsing cached profile...")
        return cached

    stats = compute_basic_stats(X, y)
    prompt = build_prompt(stats)

    llm_output = call_llm(prompt)
    llm_output = judge_profile(stats, llm_output, use_llm_judge=False)

    final_profile = {
        "raw_stats": stats,
        "llm_profile": llm_output
    }

    save_json(final_profile, f"results/profile/{dataset_name}_profile.json")

    return final_profile

def print_data_profile(profile):
    raw = profile.get("raw_stats", {})
    llm = profile.get("llm_profile", {})

    print("\nData Summary")
    print("━" * 30)

    # ----------------------------
    # Raw Stats
    # ----------------------------
    print("\nRaw Statistics")
    print(f"• Samples            : {raw.get('samples')}")
    print(f"• Features           : {raw.get('features')}")
    print(f"• Missing (%)        : {raw.get('missing_percentage')}")
    print(f"• Numerical Features : {raw.get('numerical_features')}")
    print(f"• Categorical        : {raw.get('categorical_features')}")

    # ----------------------------
    # Class Distribution
    # ----------------------------
    if raw.get("class_distribution"):
        print("\nClass Distribution")
        for cls, val in raw["class_distribution"].items():
            print(f"• {str(cls):<12} : {val * 100:.2f}%")

    # ----------------------------
    # LLM Profile
    # ----------------------------
    print("\nLLM Profile")
    print(f"• Dataset Size       : {llm.get('dataset_size', '').capitalize()}")
    print(f"• Imbalance          : {llm.get('imbalance', '').capitalize()}")
    print(f"• Missing Severity   : {llm.get('missing_severity', '').capitalize()}")
    print(f"• Feature Complexity : {llm.get('feature_complexity', '').capitalize()}")