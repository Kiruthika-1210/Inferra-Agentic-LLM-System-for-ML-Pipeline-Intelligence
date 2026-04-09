import json

from utils.llm_interface import call_llm
from agents.judge_agent import validate_strategy

def build_prompt(insights, iteration, allowed_models, prev_metrics=None):
    models_str = ", ".join(allowed_models)

    return f"""
You are an expert ML engineer.

Iteration: {iteration}

Allowed models:
{models_str}

Dataset insights:
{json.dumps(insights, indent=2)}

Previous performance:
{json.dumps(prev_metrics, indent=2) if prev_metrics else "None"}

Suggest a better ML pipeline using ONLY the allowed models.

Each pipeline must include:
- model
- reason (short)
- preprocessing steps
- hyperparameters

Return ONLY valid JSON in this format:
{{
    "model": "...",
    "reason": "...",
    "preprocessing": ["..."],
    "hyperparameters": {{}}
}}

Rules:
- Choose model ONLY from allowed_models
- Reason must be a clean single sentence (no repetition)
- Improve performance from previous iteration
- If improvement is small, try a different model
- Avoid overfitting for small datasets
- Keep preprocessing minimal
- Keep reason short (max 20 words)
- No explanation outside JSON
- hyperparameters realistic
"""
    

def generate_strategy(insights, iteration, prev_metrics=None):
    """
    LLM-driven strategy agent (core brain)
    """
    allowed_models = [
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "SVC"
    ]

    # Step 1: Prompt
    prompt = build_prompt(
        insights=insights,
        iteration=iteration,
        allowed_models=allowed_models,
        prev_metrics=prev_metrics
    )

    print(f"\n🧠 Generating strategy (Iteration {iteration})...")

    response = call_llm(prompt)

    # ✅ Fallback safety
    if not response:
        print("\n⚠️ Empty LLM response, using fallback")
        response = {"model": "RandomForest"}

    # ✅ validate
    response = validate_strategy(response)

    print("\n📊 Strategy:")
    print(json.dumps(response, indent=2))

    return response