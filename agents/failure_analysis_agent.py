import json
from utils.llm_interface import call_llm

from agents.judge_agent import clean_failure_analysis

def build_prompt(profile, insights, strategy, metrics, evaluation):
    return f"""
You are an expert ML engineer analyzing model failures.

Dataset Profile:
{json.dumps(profile, indent=2)}

Dataset Insights:
{json.dumps(insights, indent=2)}

Previous Strategy:
{json.dumps(strategy, indent=2)}

Model Performance:
{json.dumps(metrics, indent=2)}

Evaluation Result:
{json.dumps(evaluation, indent=2)}

Your task:
1. Explain WHY the model failed
2. Suggest HOW to improve the pipeline

Rules:
- Be specific (no generic advice)
- Keep explanation short (max 20 words)
- Suggest actionable changes (model, hyperparameters, preprocessing)

Return STRICT JSON:

{{
  "issue": "...",
  "reason": "...",
  "suggestion": "..."
}}
"""


def analyze_failure(profile, insights, strategy, metrics, evaluation):
    """
    LLM-driven failure analysis
    """

    # If model is already good → no need to analyze
    if evaluation.get("status") == "success":
        return {
            "issue": "none",
            "reason": "Model performs well",
            "suggestion": "No change needed"
        }

    prompt = build_prompt(profile, insights, strategy, metrics, evaluation)

    print("\n🧠 Analyzing failure...")

    response = call_llm(prompt)
    response = clean_failure_analysis(response)

    # Safety fallback
    if not response:
        return {
            "issue": evaluation.get("issue", "unknown"),
            "reason": "Fallback analysis",
            "suggestion": "Try simpler model or adjust hyperparameters"
        }

    return response