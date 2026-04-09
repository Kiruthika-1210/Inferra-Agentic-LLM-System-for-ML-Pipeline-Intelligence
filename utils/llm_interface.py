import subprocess
import json
import time
import re

import re

def clean_llm_output(output: str) -> str:
    """
    Extract and sanitize JSON from LLM output
    """

    if "{" not in output or "}" not in output:
        return ""

    start = output.find("{")
    end = output.rfind("}") + 1
    json_str = output[start:end]

    # Remove ANSI escape sequences FIRST
    json_str = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', json_str)

    # Then remove control characters
    json_str = re.sub(r'[\x00-\x1F]+', ' ', json_str)

    return json_str

def call_llm(prompt: str, retries: int = 2, delay: int = 1) -> dict:
    """
    Calls local LLM via Ollama (phi3:mini)
    Includes retry + safe parsing
    """

    for attempt in range(retries + 1):
        try:
            print("\nFetching LLM response...")

            result = subprocess.run(
                ["ollama", "run", "phi3:mini"],
                input=prompt,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="ignore"
            )
            print("\nLLM response received")

            raw_output = result.stdout.strip()
            cleaned = clean_llm_output(raw_output)

            try:
                return json.loads(cleaned)
            except Exception as e:
                print(f"\n[PARSE ERROR] Attempt {attempt+1}: {e}")

        except Exception as e:
            print(f"[LLM ERROR] Attempt {attempt+1}: {e}")

            if attempt < retries:
                print("\n Retrying...")
                time.sleep(delay)
            else:
                return {}