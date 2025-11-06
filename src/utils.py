"""
utils.py
--------
General utility functions for file management, data loading,
and pretty-printing results used throughout the F1 Predictor project.
"""

import pandas as pd
from pathlib import Path

# Define consistent data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def load_csv(file_name: str) -> pd.DataFrame:
    """
    Loads a CSV file from the data directory.

    Args:
        file_name (str): Name of the CSV file (e.g., 'features_2024.csv').

    Returns:
        pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    path = DATA_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {path}")
    print(f"📂 Loading data from: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, file_name: str):
    """
    Saves a DataFrame to the data directory.

    Args:
        df (pd.DataFrame): The data to save.
        file_name (str): Output file name.
    """
    path = DATA_DIR / file_name
    df.to_csv(path, index=False)
    print(f"💾 Saved data to: {path}")


def preview(df: pd.DataFrame, rows: int = 5):
    """
    Prints a quick summary and sample of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to preview.
        rows (int): Number of rows to display.
    """
    print(f"\n📊 Preview ({rows} rows, {df.shape[1]} columns)")
    print("-" * 60)
    print(df.head(rows))
    print("-" * 60)
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Total rows: {len(df)}\n")


def clean_driver_name(name: str) -> str:
    """
    Standardizes driver names for consistency.

    Args:
        name (str): Raw driver name (e.g., 'L. Norris' or 'Lando NORRIS').

    Returns:
        str: Cleaned full driver name.
    """
    name = name.strip().title()
    name = name.replace("  ", " ")
    return name


def check_data_pipeline():
    """
    Quick sanity check to confirm that all expected files exist.
    """
    expected_files = [
        "race_results_2024.csv",
        "features_2024.csv",
        "xgb_model.joblib"
    ]
    print("🔍 Checking data pipeline...")
    for f in expected_files:
        path = DATA_DIR / f
        if path.exists():
            print(f"✅ Found: {f}")
        else:
            print(f"⚠️ Missing: {f} — generate it using the appropriate script.")
    print("")


if __name__ == "__main__":
    # Example usage for testing utils
    check_data_pipeline()
