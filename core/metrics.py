from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_pipeline_complexity(pipeline):
    """
    Simple complexity score
    """
    steps = len(pipeline.steps)

    hyperparams = 0
    model = pipeline.steps[-1][1]

    if hasattr(model, "get_params"):
        hyperparams = len(model.get_params())

    return {
        "num_steps": steps,
        "num_hyperparameters": hyperparams
    }


def compute_metrics(y_train, y_test, train_pred, test_pred, runtime, peak_memory, pipeline):
    """
    Compute full evaluation metrics
    """

    # 🔥 Detect classification type
    num_classes = len(set(y_test))
    avg = "binary" if num_classes == 2 else "weighted"

    # Accuracy
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    # Classification metrics
    precision = precision_score(y_test, test_pred, average=avg, zero_division=0)
    recall = recall_score(y_test, test_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, test_pred, average=avg, zero_division=0)

    # Complexity
    complexity = compute_pipeline_complexity(pipeline)

    return {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "runtime": runtime,
        "peak_memory_kb": round(peak_memory / 1024, 2),  # 🔥 FIXED
        "pipeline_complexity": complexity
    }