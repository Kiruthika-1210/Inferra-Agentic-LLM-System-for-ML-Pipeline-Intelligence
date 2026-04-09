import pandas as pd


def preprocess_data(input_path, output_path, target_column):
    """
    Basic structural preprocessing:
    - Load dataset
    - Clean column names
    - Remove empty rows
    - Validate target column
    - Fix simple datatype issues
    """

    # Load dataset
    if input_path.startswith("http://") or input_path.startswith("https://"):
        print("[Preprocess] Loading from URL...")
    else:
        print("[Preprocess] Loading from local file...")

    try:
        df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )

    print("Original Shape:", df.shape)
    
    # Remove completely empty rows
    df.dropna(how="all", inplace=True)

    # Normalize target column name
    target_column = target_column.strip().lower().replace(" ", "_")

    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    # Convert numeric-like columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            continue
    
     # Ensure categorical columns are strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    print("Processed Shape:", df.shape)
    
    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print(f"Preprocessing complete → {output_path}")

    return target_column   