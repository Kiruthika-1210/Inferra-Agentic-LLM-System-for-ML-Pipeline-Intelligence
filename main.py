import argparse
import os
import json
from urllib.parse import urlparse

from data.preprocess import preprocess_data
from data.data_loader import load_data
from profiling.data_profiler import profile_data
from agents.dataset_analyzer_agent import analyze_dataset


def main():
    # ----------------------------
    # Argument Parser
    # ----------------------------
    parser = argparse.ArgumentParser(description="Auto ML Pipeline System")

    parser.add_argument("--file", type=str, required=True, help="Path or URL to input dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["profile", "analyze", "all"],
        help="Pipeline stage to execute"
    )

    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ✅ Fix path issue
    input_file = os.path.join(BASE_DIR, args.file)
    target_column = args.target

    # ----------------------------
    # Extract dataset name
    # ----------------------------
    if input_file.startswith("http"):
        parsed_url = urlparse(input_file)
        dataset_name = os.path.splitext(os.path.basename(parsed_url.path))[0]
    else:
        dataset_name = os.path.splitext(os.path.basename(input_file))[0]

    if dataset_name == "":
        dataset_name = "dataset"

    # ----------------------------
    # Output path
    # ----------------------------
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    cleaned_file = os.path.join(processed_dir, f"{dataset_name}_cleaned.csv")

    # ----------------------------
    # STEP 1: Preprocessing
    # ----------------------------
    target_column = preprocess_data(
        input_path=input_file,
        output_path=cleaned_file,
        target_column=target_column
    )

    # ----------------------------
    # STEP 2: Load + Split
    # ----------------------------
    X_train, X_test, y_train, y_test = load_data(
        file_path=cleaned_file,
        target_column=target_column
    )

    print("\n📊 Dataset Shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # ----------------------------
    # STEP 3: Profiling
    # ----------------------------
    profile = profile_data(X_train, y_train)

    if not profile.get("llm_profile"):
        print("\nLLM profiling failed, using fallback.")

    if args.stage == "profile":
        print("\nProfiling stage completed.")
        return

    # ----------------------------
    # STEP 4: Analysis
    # ----------------------------
    insights = analyze_dataset(profile)

    if not insights:
        print("\nInsights generation failed")

    print("\n📊 Final Insights:")
    print(json.dumps(insights, indent=2))

    if args.stage == "analyze":
        print("\nAnalysis stage completed.")
        return


if __name__ == "__main__":
    main()