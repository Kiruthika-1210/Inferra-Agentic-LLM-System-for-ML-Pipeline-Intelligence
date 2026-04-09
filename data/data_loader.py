import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path, target_column, test_size=0.2, random_state=42):
    """
    Loads dataset and splits into train/test sets
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() < 20 else None
    )

    print("Data loaded and split")

    return X_train, X_test, y_train, y_test