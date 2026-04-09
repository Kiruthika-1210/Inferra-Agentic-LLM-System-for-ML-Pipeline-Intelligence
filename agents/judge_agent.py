import json
from utils.llm_interface import call_llm


# ----------------------------
# PROFILE VALIDATION
# ----------------------------
def rule_based_validation(stats: dict, llm_profile: dict) -> dict:
    """
    Deterministic validation of LLM output
    """

    # --- Imbalance ---
    dist = stats.get("class_distribution", {})
    
    if len(dist) == 2:
        values = list(dist.values())
        ratio = min(values) / max(values)

        if ratio < 0.5:
            llm_profile["imbalance"] = "high"
        elif ratio < 0.75:
            llm_profile["imbalance"] = "moderate"
        else:
            llm_profile["imbalance"] = "low"

    # --- Missing ---
    missing = stats.get("missing_percentage", 0)

    if missing < 5:
        llm_profile["missing_severity"] = "low"
    elif missing < 20:
        llm_profile["missing_severity"] = "moderate"
    else:
        llm_profile["missing_severity"] = "high"

    # --- Feature type ---
    if stats.get("categorical_features", 0) == 0:
        llm_profile["feature_complexity"] = "numerical"
    elif stats.get("numerical_features", 0) == 0:
        llm_profile["feature_complexity"] = "categorical"
    else:
        llm_profile["feature_complexity"] = "mixed"

    return llm_profile


def llm_judge(stats: dict, llm_profile: dict) -> dict:
    """
    Optional LLM-based validation
    """

    prompt = f"""
You are a strict validator.

Given dataset statistics:
{json.dumps(stats, indent=2)}

And an LLM-generated profile:
{json.dumps(llm_profile, indent=2)}

Check correctness and fix if needed.

Return ONLY valid JSON.
"""

    judged = call_llm(prompt)
    return judged if judged else llm_profile


def judge_profile(stats: dict, llm_profile: dict, use_llm_judge: bool = False) -> dict:
    """
    Profile validation pipeline
    """

    validated = rule_based_validation(stats, llm_profile)

    if use_llm_judge:
        validated = llm_judge(stats, validated)

    return validated


# ----------------------------
# INSIGHT CLEANING
# ----------------------------
def clean_insights(data: dict) -> dict:
    """
    Clean LLM-generated insights
    """

    cleaned = []

    for item in data.get("insights", []):
        if not isinstance(item, str):
            continue

        item = item.replace("[K", "").strip()
        item = item[:80]

        if item:
            cleaned.append(item)

    return {"insights": cleaned}


# ----------------------------
# STRATEGY VALIDATION (SINGLE PIPELINE)
# ----------------------------
def validate_strategy(strategy: dict) -> dict:
    """
    Validate single pipeline strategy
    """

    valid_models = [
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "SVC"
    ]

    # --- Model ---
    if strategy.get("model") not in valid_models:
        strategy["model"] = "RandomForest"

    # --- Reason ---
    if not strategy.get("reason"):
        strategy["reason"] = "default model selection"

    # --- Preprocessing ---
    if not isinstance(strategy.get("preprocessing"), list):
        strategy["preprocessing"] = []

    # --- Hyperparameters ---
    if not isinstance(strategy.get("hyperparameters"), dict):
        strategy["hyperparameters"] = {}

    # --- Defaults ---
    if strategy["model"] == "RandomForest":
        strategy["hyperparameters"].setdefault("n_estimators", 100)
        strategy["hyperparameters"].setdefault("class_weight", "balanced")

    return strategy