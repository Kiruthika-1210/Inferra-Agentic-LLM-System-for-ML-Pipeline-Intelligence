import time
import traceback
import tracemalloc


def run_pipeline(pipeline, X_train, X_test, y_train, y_test):
    """
    Train model and return predictions + runtime + memory
    """

    try:
        # 🔥 START memory tracking BEFORE training
        tracemalloc.start()

        start_time = time.time()

        # Train
        pipeline.fit(X_train, y_train)

        # Predictions
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        runtime = time.time() - start_time

        # 🔥 GET memory AFTER execution
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "success": True,
            "train_pred": train_pred,
            "test_pred": test_pred,
            "runtime": round(runtime, 4),
            "peak_memory": peak,   
            "pipeline": pipeline
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }