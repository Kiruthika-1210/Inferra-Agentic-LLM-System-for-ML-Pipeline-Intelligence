import json

from utils.llm_interface import call_llm
from agents.judge_agent import validate_strategy


def normalize_preprocessing(steps):
    normalized = []

    for step in steps:
        s = step.lower()

        if "smote" in s:
            normalized.append("SMOTE")
        elif "scale" in s or "standard" in s:
            normalized.append("StandardScaler")
        elif "normalize" in s:
            normalized.append("Normalizer")

    return list(set(normalized))


def sanitize_hyperparameters(model, params):
    allowed = {
        "LogisticRegression": {"C", "max_iter"},
        "RandomForest": {"n_estimators", "max_depth", "min_samples_split"},
        "GradientBoosting": {"n_estimators", "learning_rate", "max_depth", "min_samples_split"},
        "SVC": {"C", "kernel", "gamma"}
    }

    valid_keys = allowed.get(model, set())
    cleaned = {}

    for k, v in params.items():
        if k in valid_keys:

            # 🔥 FIX 1: skip None values
            if v is None:
                continue

            # 🔥 FIX 2: handle list values
            if isinstance(v, list):
                cleaned[k] = v[0]
            else:
                cleaned[k] = v

    return cleaned


def build_prompt(insights, iteration, allowed_models,
                 prev_metrics=None,
                 failure_reason=None,
                 refinement=None):

    models_str = ", ".join(allowed_models)

    return f"""
You are an expert ML engineer designing ML pipelines.

Iteration: {iteration}

Allowed models:
{models_str}

Dataset insights:
{json.dumps(insights, indent=2)}

Previous performance:
{json.dumps(prev_metrics, indent=2) if prev_metrics else "None"}

Failure from previous iteration:
{failure_reason if failure_reason else "None"}

Refinement suggestions:
{refinement if refinement else "None"}

Your task:
Suggest an improved ML pipeline.

Requirements:
- Use ONLY allowed models
- Improve performance based on previous metrics
- Address failure reason if provided
- Apply refinement suggestions if provided
- Avoid overfitting
- Keep preprocessing minimal and valid

Preprocessing MUST be chosen only from:
["SMOTE", "StandardScaler", "Normalizer"]

Return STRICT JSON ONLY:

{{
  "model": "...",
  "reason": "...",
  "preprocessing": ["..."],
  "hyperparameters": {{}},
  "confidence": 0.0
}}

Rules:
- No text outside JSON
- All keys must use double quotes
- No trailing commas
- Reason: max 20 words
- Confidence: 0.0 to 1.0
- Hyperparameters must be valid for chosen model
"""


def generate_strategy(insights, iteration,
                      prev_metrics=None,
                      failure_reason=None,
                      refinement=None):

    allowed_models = [
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
        "SVC"
    ]

    prompt = build_prompt(
        insights=insights,
        iteration=iteration,
        allowed_models=allowed_models,
        prev_metrics=prev_metrics,
        failure_reason=failure_reason,
        refinement=refinement
    )

    print(f"\n🧠 Generating strategy (Iteration {iteration})...")

    response = call_llm(prompt)

    # 🔧 FIX 1: Ensure dict
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except:
            print("\n⚠️ Failed to parse LLM response, using fallback")
            response = {}

    # 🔧 FIX 2: Fallback if empty
    if not response:
        response = {
            "model": "RandomForest",
            "reason": "Fallback safe model",
            "preprocessing": [],
            "hyperparameters": {},
            "confidence": 0.5
        }

    # 🔧 FIX 3: Ensure required keys
    response.setdefault("model", "RandomForest")
    response.setdefault("reason", "Fallback reason")
    response.setdefault("preprocessing", [])
    response.setdefault("hyperparameters", {})

    # Validate
    response = validate_strategy(response)

    # Normalize preprocessing
    response["preprocessing"] = normalize_preprocessing(
        response.get("preprocessing", [])
    )

    # Sanitize hyperparameters
    response["hyperparameters"] = sanitize_hyperparameters(
        response.get("model"),
        response.get("hyperparameters", {})
    )

    # 🔧 FIX 4: Confidence safety
    try:
        response["confidence"] = float(response.get("confidence", 0.7))
    except:
        response["confidence"] = 0.7

    print("\n📊 Strategy:")
    print(json.dumps(response, indent=2))

    return response