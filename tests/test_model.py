"""
tests/test_model.py
-------------------
Basic unit tests for the F1 Predictor project.
Uses pytest to confirm that data, features, and model artifacts exist
and that the model can make predictions without errors.
"""

import pandas as pd
import joblib
from pathlib import Path

from src.feature_engineering import build_features
from src.model_training import train_model
from src.utils import load_csv, check_data_pipeline

DATA_DIR = Path("data")


def test_data_files_exist():
    """Ensure that the essential data files are present."""
    check_data_pipeline()
    for filename in [
        "race_results_2024.csv",
        "features_2024.csv",
        "xgb_model.joblib",
    ]:
        assert (DATA_DIR / filename).exists(), f"{filename} missing."


def test_feature_structure():
    """Validate that the features CSV has expected columns."""
    df = pd.read_csv(DATA_DIR / "features_2024.csv")
    expected = {
        "driver",
        "constructor",
        "avg_grid",
        "avg_finish",
        "races",
        "total_points",
        "form_score",
    }
    assert expected.issubset(df.columns), "Feature columns incomplete."
    assert not df.isna().any().any(), "Found NaN values in feature set."


def test_model_training_and_prediction():
    """Train model and confirm predictions run without crashing."""
    model = train_model(2024)
    df = pd.read_csv(DATA_DIR / "features_2024.csv")
    preds = model.predict(df[["avg_grid", "races", "total_points", "finish_diff", "form_score"]])
    assert len(preds) == len(df), "Prediction output length mismatch."
    assert preds.min() >= 0, "Predicted finish positions invalid (negative values)."


def test_build_features_runs_clean():
    """Rebuild features to confirm script executes end-to-end."""
    df = build_features(2024)
    assert isinstance(df, pd.DataFrame)
    assert "driver" in df.columns
    assert len(df) > 0, "No feature rows generated."
