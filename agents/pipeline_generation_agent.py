from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from collections import Counter

def is_imbalanced(y, threshold=0.6):
    counts = Counter(y)
    total = sum(counts.values())

    max_ratio = max(counts.values()) / total

    return max_ratio > threshold

def build_preprocessor(X, strategy):
    """
    Hybrid preprocessing:
    - Mandatory: handle missing values + categorical encoding
    - LLM-guided: scaling/normalization
    """

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    transformers = []

    # ----------------------------
    # Numeric pipeline
    # ----------------------------
    if numerical_cols:

        num_steps = []

        # 🔥 MUST: handle missing values
        num_steps.append(("imputer", SimpleImputer(strategy="mean")))

        # LLM-guided scaling
        if "StandardScaler" in strategy.get("preprocessing", []):
            num_steps.append(("scaler", StandardScaler()))
        elif "Normalizer" in strategy.get("preprocessing", []):
            num_steps.append(("normalizer", Normalizer()))

        num_pipeline = Pipeline(num_steps)
        transformers.append(("num", num_pipeline, numerical_cols))

    # ----------------------------
    # Categorical pipeline
    # ----------------------------
    if categorical_cols:

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),  # 🔥 MUST
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(("cat", cat_pipeline, categorical_cols))

    if not transformers:
        return None

    return ColumnTransformer(transformers, remainder="drop")


def get_model(model_name, params):
    """
    Map model name to sklearn model
    """

    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, **params)

    elif model_name == "RandomForest":
        return RandomForestClassifier(**params)

    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(**params)

    elif model_name == "SVC":
        return SVC(**params)

    else:
        return RandomForestClassifier()  # fallback


def generate_pipeline(strategy: dict, X_train, y_train):

    use_smote = "SMOTE" in strategy.get("preprocessing", [])

    num_classes = len(set(y_train))
    imbalanced = is_imbalanced(y_train)

    if use_smote and not imbalanced:
        use_smote = False

    # ----------------------------
    # Model
    # ----------------------------
    model = get_model(
        strategy.get("model"),
        strategy.get("hyperparameters", {})
    )

    # ----------------------------
    # Preprocessor
    # ----------------------------
    preprocessor = build_preprocessor(X_train, strategy)

    # ----------------------------
    # Pipeline
    # ----------------------------
    steps = []

    if preprocessor:
        steps.append(("preprocessor", preprocessor))

    if use_smote:
        steps.append(("smote", SMOTE()))

    steps.append(("model", model))

    # Choose pipeline type
    if use_smote:
        pipeline = ImbPipeline(steps)
    else:
        pipeline = Pipeline(steps)

    return pipeline