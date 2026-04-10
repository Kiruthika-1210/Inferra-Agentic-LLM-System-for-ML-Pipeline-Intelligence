import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from core.metrics import compute_metrics
from core.execution_engine import run_pipeline


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])


def run_manual_baseline(df, target, dataset_name, save_path):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])

    execution = run_pipeline(pipeline, X_train, X_test, y_train, y_test)

    metrics = compute_metrics(
        y_train,
        y_test,
        execution["train_pred"],
        execution["test_pred"],
        execution["runtime"],
        execution["peak_memory"],
        execution["pipeline"]
    )

    row = {
        "dataset": dataset_name,
        "model": "RandomForest",
        **metrics
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame([row]).to_csv(save_path, index=False)


def run_gridsearch_baseline(df, target, dataset_name, save_path):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3)

    start = time.time()
    grid.fit(X_train, y_train)
    runtime = time.time() - start

    best_model = grid.best_estimator_

    execution = run_pipeline(best_model, X_train, X_test, y_train, y_test)

    metrics = compute_metrics(
        y_train,
        y_test,
        execution["train_pred"],
        execution["test_pred"],
        runtime,
        execution["peak_memory"],
        execution["pipeline"]
    )

    row = {
        "dataset": "dataset",
        "model": "GridSearch_RF",
        "best_params": str(grid.best_params_),
        **metrics
    }

    pd.DataFrame([row]).to_csv(save_path, index=False)