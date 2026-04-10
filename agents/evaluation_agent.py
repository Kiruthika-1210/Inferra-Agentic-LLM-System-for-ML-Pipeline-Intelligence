def evaluate_results(metrics: dict, execution_success: bool = True) -> dict:
    """
    Evaluate model performance including execution failures
    """

    # ----------------------------
    # Handle execution failure FIRST
    # ----------------------------
    if not execution_success:
        return {
            "status": "fail",
            "issue": "execution_failure",
            "reason": "Pipeline execution failed"
        }

    train_acc = metrics.get("train_accuracy", 0)
    test_acc = metrics.get("test_accuracy", 0)

    gap = train_acc - test_acc

    result = {
        "status": "ok",
        "issue": None,
        "reason": None
    }

    # ----------------------------
    # Overfitting
    # ----------------------------
    if gap > 0.1:
        result["status"] = "fail"
        result["issue"] = "overfitting"
        result["reason"] = f"High train-test gap ({gap:.2f})"

    # ----------------------------
    # Underfitting
    # ----------------------------
    elif train_acc < 0.75 and test_acc < 0.75:
        result["status"] = "fail"
        result["issue"] = "underfitting"
        result["reason"] = "Both train and test accuracy are low"

    # ----------------------------
    # Good fit
    # ----------------------------
    else:
        result["status"] = "success"
        result["issue"] = "good_fit"
        result["reason"] = "Model generalizes well"

    return result