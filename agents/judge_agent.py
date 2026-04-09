import json
from utils.llm_interface import call_llm


def rule_based_validation(stats: dict, llm_profile: dict) -> dict:
    """
    Deterministic validation of LLM output
    """

    # --- Imbalance Correction ---
    dist = stats.get("class_distribution", {})
    
    if len(dist) == 2:
        values = list(dist.values())
        ratio = min(values) / max(values)

        if ratio < 0.5:
            correct = "high"
        elif ratio < 0.75:
            correct = "moderate"
        else:
            correct = "low"

        llm_profile["imbalance"] = correct

    # --- Missing Values ---
    missing = stats.get("missing_percentage", 0)

    if missing == 0:
        llm_profile["missing_severity"] = "low"
    elif missing < 5:
        llm_profile["missing_severity"] = "low"
    elif missing < 20:
        llm_profile["missing_severity"] = "moderate"
    else:
        llm_profile["missing_severity"] = "high"

    # --- Feature Type ---
    if stats.get("categorical_features", 0) == 0:
        llm_profile["feature_complexity"] = "numerical"
    elif stats.get("numerical_features", 0) == 0:
        llm_profile["feature_complexity"] = "categorical"
    else:
        llm_profile["feature_complexity"] = "mixed"

    return llm_profile


def llm_judge(stats: dict, llm_profile: dict) -> dict:
    """
    Optional LLM-based validation (secondary judge)
    """

    prompt = f"""
You are a strict validator.

Given dataset statistics:
{json.dumps(stats, indent=2)}

And an LLM-generated profile:
{json.dumps(llm_profile, indent=2)}

Check if the profile is correct.
If incorrect, fix it.

Return ONLY valid JSON in same format.
"""

    judged = call_llm(prompt)

    return judged if judged else llm_profile


def judge_profile(stats: dict, llm_profile: dict, use_llm_judge: bool = False) -> dict:
    """
    Main judge pipeline
    """

    # Step 1: Rule-based correction
    validated = rule_based_validation(stats, llm_profile)

    # Step 2: Optional LLM judge
    if use_llm_judge:
        validated = llm_judge(stats, validated)

    return validated

def clean_insights(insights: dict) -> dict:
    """
    Light validation for insights
    """

    cleaned = []

    for item in insights.get("insights", []):
        if not isinstance(item, str):
            continue

        # Remove weird characters
        item = item.replace("[K", "").strip()

        # Limit length
        item = item[:80]

        # Ensure not empty
        if item:
            cleaned.append(item)

    return {"insights": cleaned}