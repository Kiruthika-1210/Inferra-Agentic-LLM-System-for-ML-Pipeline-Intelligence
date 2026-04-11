import argparse
import os
import json
import pandas as pd
from urllib.parse import urlparse

from data.preprocess import preprocess_data
from data.data_loader import load_data
from profiling.data_profiler import profile_data, print_data_profile
from agents.dataset_analyzer_agent import analyze_dataset
from agents.strategy_agent import generate_strategy
from agents.pipeline_generation_agent import generate_pipeline
from core.execution_engine import run_pipeline
from core.metrics import compute_metrics
from agents.evaluation_agent import evaluate_results
from agents.failure_analysis_agent import analyze_failure
from experiments.logger import save_log


def print_iteration(strategy, metrics):
    print("\nStrategy")
    print(f"• Model          : {strategy.get('model')}")
    print(f"• Reason         : {strategy.get('reason', 'N/A')}")
    print(f"• Preprocessing  : {', '.join(strategy.get('preprocessing', [])) or 'None'}")
    print(f"• Confidence     : {strategy.get('confidence', 'N/A')}")

    if strategy.get("hyperparameters"):
        print("\nHyperparameters")
        for k, v in strategy["hyperparameters"].items():
            print(f"• {k:<18}: {v}")

    print("\nPerformance")
    print(f"• Train Accuracy : {metrics.get('train_accuracy', 0):.4f}")
    print(f"• Test Accuracy  : {metrics.get('test_accuracy', 0):.4f}")
    print(f"• Precision      : {metrics.get('precision', 0):.4f}")
    print(f"• Recall         : {metrics.get('recall', 0):.4f}")
    print(f"• F1 Score       : {metrics.get('f1_score', 0):.4f}")

    print("\nEfficiency")
    print(f"• Runtime        : {metrics.get('runtime', 0):.2f} sec")
    print(f"• Memory         : {metrics.get('peak_memory_kb', 0):.2f} KB")

    if metrics.get("pipeline_complexity"):
        comp = metrics["pipeline_complexity"]
        print("\nPipeline Complexity")
        print(f"• Steps          : {comp.get('num_steps')}")
        print(f"• Hyperparameters: {comp.get('num_hyperparameters')}")

def clean_text(text):
    return " ".join(str(text).split())

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

    print("\n" + "━"*50)
    print(f"DATASET: {dataset_name.upper()}")
    print("━"*50)

    print("\nDataset Shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # ----------------------------
    # Profiling + Analysis
    # ----------------------------
    profile = profile_data(X_train, y_train, dataset_name)
    print_data_profile(profile)

    insights = analyze_dataset(profile, dataset_name)

    print("\nKey Insights:")
    for insight in insights.get("insights", []):
        print(f"• {clean_text(insight)}")

    if insights.get("risk_factors"):
        print("\nRisk Factors")
        for risk in insights.get("risk_factors", []):
            print(f"• {clean_text(risk)}")

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
    best_metrics = None

    iterations_log = []

    for iteration in range(1, max_iterations + 1):
        print("\n" + "━"*50)
        print(f"ITERATION {iteration}")
        print("━"*50)

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
            print("\nExecution failed:", execution_output["error"])

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
            execution_output["peak_memory"],   
            execution_output["pipeline"]
        )

        print_iteration(strategy, metrics)

        # ----------------------------
        # Evaluation
        # ----------------------------
        evaluation = evaluate_results(metrics, execution_success=True)

        status = "Good Fit" if evaluation["status"] == "success" else f"{evaluation['issue'].capitalize()}"

        print("\nEvaluation")
        print(f"• Status         : {status}")
        print(f"• Reason         : {evaluation.get('reason', 'N/A')}")

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

        if evaluation["status"] == "fail":
            print("\nRefinement Suggestion")
            print(f"• {clean_text(failure.get('suggestion', 'No suggestion'))}")

        # ----------------------------
        # Update failure signal
        # ----------------------------
        if evaluation["status"] == "fail":
            failure_reason = evaluation["issue"]
            refinement = failure.get("suggestion")
        else:
            failure_reason = None
            refinement = None

        iterations_log.append({
            "iteration": iteration,
            "strategy": strategy,
            "metrics": metrics,
            "evaluation": evaluation,
            "failure_reason": failure_reason,
            "refinement": refinement
        })

        # ----------------------------
        # Stopping Criteria
        # ----------------------------
        current_accuracy = metrics.get("test_accuracy", 0)

        # Update best accuracy
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_strategy = strategy
            best_metrics = metrics

        # 1️⃣ Desired accuracy reached
        if current_accuracy >= target_accuracy:
            print("\nTarget accuracy reached. Stopping early.")
            break

        # 2️⃣ Improvement check
        if prev_metrics:
            prev_acc = prev_metrics.get("test_accuracy", 0)
            improvement = current_accuracy - prev_acc

            if improvement < improvement_threshold:
                print("\nImprovement below threshold — stopping early")
                break
        
        prev_metrics = metrics


    print("\n" + "━"*50)
    print("🏆 FINAL RESULT")
    print("━"*50)

    # ----------------------------
    # Strategy
    # ----------------------------
    print("\n🧠 Best Strategy")
    print(f"• Model          : {best_strategy.get('model')}")
    print(f"• Reason         : {best_strategy.get('reason', 'N/A')}")
    print(f"• Preprocessing  : {', '.join(best_strategy.get('preprocessing', [])) or 'None'}")
    print(f"• Confidence     : {best_strategy.get('confidence', 'N/A')}")

    # ----------------------------
    # Hyperparameters
    # ----------------------------
    if best_strategy.get("hyperparameters"):
        print("\n⚙️ Hyperparameters")
        for k, v in best_strategy["hyperparameters"].items():
            print(f"• {k:<20}: {v}")

    # ----------------------------
    # Metrics
    # ----------------------------
    print("\n📈 Best Performance")
    print(f"• Train Accuracy : {best_metrics.get('train_accuracy', 0):.4f}")
    print(f"• Test Accuracy  : {best_metrics.get('test_accuracy', 0):.4f}")
    print(f"• Precision      : {best_metrics.get('precision', 0):.4f}")
    print(f"• Recall         : {best_metrics.get('recall', 0):.4f}")
    print(f"• F1 Score       : {best_metrics.get('f1_score', 0):.4f}")

    # ----------------------------
    # Efficiency
    # ----------------------------
    print("\n⚡ Efficiency")
    print(f"• Runtime        : {best_metrics.get('runtime', 0):.2f} sec")
    print(f"• Memory         : {best_metrics.get('peak_memory_kb', 0):.2f} KB")

    # ----------------------------
    # Complexity
    # ----------------------------
    if best_metrics.get("pipeline_complexity"):
        comp = best_metrics["pipeline_complexity"]
        print("\n🧩 Pipeline Complexity")
        print(f"• Steps          : {comp.get('num_steps')}")
        print(f"• Hyperparameters: {comp.get('num_hyperparameters')}")

    # ----------------------------
    # Iterations
    # ----------------------------
    print(f"\n🔁 Total Iterations : {iteration}")

    stopping_reason = "max_iterations"
        
    if best_accuracy >= target_accuracy:
        stopping_reason = "target_accuracy"
    elif prev_metrics:
        prev_acc = prev_metrics.get("test_accuracy", 0)
        if abs(best_accuracy - prev_acc) < improvement_threshold:
            stopping_reason = "low_improvement"

    save_log(
        dataset_name=dataset_name,
        profile=profile,
        insights=insights,
        iterations=iterations_log,
        final_result={
            "best_strategy": best_strategy,
            "best_metrics": best_metrics,   # 🔥 ADD THIS
            "best_accuracy": best_accuracy,
            "total_iterations": iteration,
            "stopping_reason": stopping_reason
        }
    )

    # ----------------------------
    # SAVE AGENTIC RESULTS (FIXED)
    # ----------------------------
    results_dir = os.path.join(BASE_DIR, "experiments", "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    agentic_path = os.path.join(results_dir, "agentic.csv")

    result_row = {
        "dataset": dataset_name,
        "model": best_strategy.get("model"),

        "train_accuracy": best_metrics.get("train_accuracy"),
        "test_accuracy": best_metrics.get("test_accuracy"),
        "precision": best_metrics.get("precision"),
        "recall": best_metrics.get("recall"),
        "f1_score": best_metrics.get("f1_score"),

        "runtime": best_metrics.get("runtime"),
        "peak_memory_kb": best_metrics.get("peak_memory_kb"),

        "pipeline_complexity": str(best_metrics.get("pipeline_complexity"))
    }

    df = pd.DataFrame([result_row])
    df.to_csv(agentic_path, index=False)

if __name__ == "__main__":
    main()

    