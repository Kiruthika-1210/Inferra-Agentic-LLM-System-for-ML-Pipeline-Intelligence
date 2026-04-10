import argparse
import os
import json
from urllib.parse import urlparse

from data.preprocess import preprocess_data
from data.data_loader import load_data
from profiling.data_profiler import profile_data
from agents.dataset_analyzer_agent import analyze_dataset
from agents.strategy_agent import generate_strategy
from agents.pipeline_generation_agent import generate_pipeline
from core.execution_engine import run_pipeline
from core.metrics import compute_metrics
from agents.evaluation_agent import evaluate_results
from agents.failure_analysis_agent import analyze_failure


def main():

    parser = argparse.ArgumentParser(description="Agentic AutoML System")

    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)

    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 🔥 FIX: handle URL vs local file
    if args.file.startswith("http"):
        input_file = args.file
    else:
        input_file = os.path.join(BASE_DIR, args.file)

    # ----------------------------
    # Dataset name
    # ----------------------------
    if input_file.startswith("http"):
        parsed_url = urlparse(input_file)
        dataset_name = os.path.splitext(os.path.basename(parsed_url.path))[0]
    else:
        dataset_name = os.path.splitext(os.path.basename(input_file))[0]

    if not dataset_name:
        dataset_name = "dataset"

    # ----------------------------
    # Preprocess
    # ----------------------------
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    cleaned_file = os.path.join(processed_dir, f"{dataset_name}_cleaned.csv")

    target_column = preprocess_data(
        input_path=input_file,
        output_path=cleaned_file,
        target_column=args.target
    )

    # ----------------------------
    # Load Data
    # ----------------------------
    X_train, X_test, y_train, y_test = load_data(
        file_path=cleaned_file,
        target_column=target_column
    )

    print("\n📊 Dataset Shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # ----------------------------
    # Profiling + Analysis
    # ----------------------------
    profile = profile_data(X_train, y_train)
    insights = analyze_dataset(profile)

    print("\n📊 Insights:")
    print(json.dumps(insights, indent=2))

    # ----------------------------
    # ITERATIVE LOOP
    # ----------------------------
    max_iterations = 3
    prev_metrics = None
    failure_reason = None
    refinement = None
    target_accuracy = 0.85
    improvement_threshold = 0.005  # 0.5%
    best_strategy = None
    best_accuracy = float("-inf")

    for iteration in range(1, max_iterations + 1):

        print(f"\n🚀 Iteration {iteration}")

        # ----------------------------
        # Strategy Agent
        # ----------------------------
        strategy = generate_strategy(
            insights=insights,
            iteration=iteration,
            prev_metrics=prev_metrics,
            failure_reason=failure_reason,
            refinement=refinement
        )

        # ----------------------------
        # 🔜 Pipeline Generation (next step)
        # ----------------------------
        pipeline = generate_pipeline(strategy, X_train, y_train)

        execution_output = run_pipeline(
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        if not execution_output["success"]:
            print("\n❌ Execution failed:", execution_output["error"])

            failure_reason = "Execution failure"
            refinement = "Simplify model or fix parameters"
            continue

        # ----------------------------
        # Metrics
        # ----------------------------
        metrics = compute_metrics(
            y_train,
            y_test,
            execution_output["train_pred"],
            execution_output["test_pred"],
            execution_output["runtime"],
            execution_output["peak_memory"],   # 🔥 NEW
            execution_output["pipeline"]
        )

        print("\n📈 Metrics:")
        print(json.dumps(metrics, indent=2))

        # ----------------------------
        # Stopping Criteria
        # ----------------------------
        current_accuracy = metrics.get("test_accuracy", 0)

        # Update best accuracy
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_strategy = strategy

        # 1️⃣ Desired accuracy reached
        if current_accuracy >= target_accuracy:
            print("\n🎯 Target accuracy reached. Stopping early.")
            break

        # 2️⃣ Improvement check
        if prev_metrics:
            prev_acc = prev_metrics.get("test_accuracy", 0)
            improvement = current_accuracy - prev_acc

            if improvement < improvement_threshold:
                print("\n📉 Improvement too small. Stopping early.")
                break

        # ----------------------------
        # Evaluation
        # ----------------------------
        evaluation = evaluate_results(metrics, execution_success=True)

        print("\n🧠 Evaluation:")
        print(json.dumps(evaluation, indent=2))

        # ----------------------------
        # Failure Analysis
        # ----------------------------
        failure = analyze_failure(
            profile,
            insights,
            strategy,
            metrics,
            evaluation
        )

        print("\n🧠 Failure Analysis:")
        print(json.dumps(failure, indent=2))

        # ----------------------------
        # Update failure signal
        # ----------------------------
        if evaluation["status"] == "fail":
            failure_reason = evaluation["issue"]
            refinement = failure.get("suggestion")
        else:
            failure_reason = None
            refinement = None

        prev_metrics = metrics

    print("\n🏆 FINAL BEST STRATEGY:")
    print(json.dumps(best_strategy, indent=2))

    print(f"\n🏆 FINAL BEST ACCURACY: {best_accuracy}")

if __name__ == "__main__":
    main()