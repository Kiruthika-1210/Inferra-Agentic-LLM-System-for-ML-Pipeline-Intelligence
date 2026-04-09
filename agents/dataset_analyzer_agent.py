import json
import os

from utils.llm_interface import call_llm
from agents.judge_agent import clean_insights

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_prompt(profile: dict) -> str:
    """
    Build prompt for dataset analysis
    """

    return f"""
You are an expert machine learning engineer.

Given the dataset profile:
{json.dumps(profile, indent=2)}

Identify key machine learning challenges and considerations.

Focus on:
- data size
- class imbalance
- feature types
- missing values

Rules:
- insights must be short (max 12 words)
- each insight must be one sentence
- risk_factors must be ML problems (overfitting, imbalance, etc.)
- no explanations
- no special characters
- strictly valid JSON
- no line breaks

Return ONLY valid JSON in this format:
{{
  "insights": [
    "...",
    "...",
    "..."
  ],
  "risk_factors": [
  "...", 
  "..."
  ]
}}

Do not include newlines inside strings.
Do not include explanations.
Ensure JSON is strictly valid.
"""


def analyze_dataset(profile: dict) -> dict:
    """
    LLM-driven dataset analysis agent with caching
    """

    # Step 0: Check cache
    cache_path = "results/insights.json"
    cached = load_json(cache_path)

    if cached:
        print("\nUsing cached insights...")
        return cached

    # Step 1: Build prompt
    prompt = build_prompt(profile)

    print("\nAnalyzing dataset...")

    # Step 2: LLM call
    response = call_llm(prompt)

    # Light judge
    response = clean_insights(response)

    # Step 3: Safety check
    if not response.get("insights"):
        print("\nLLM returned empty insights")

    # Step 4: Save result
    save_json(response, cache_path)

    print("\nDataset analysis complete")

    print("\nDataset Insights:")
    print(json.dumps(response, indent=2))

    return response