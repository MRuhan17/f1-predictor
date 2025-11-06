"""
model_training.py
-----------------
Trains an XGBoost regression model to predict average finishing positions
based on driver and constructor performance features.
Evaluates accuracy and saves the trained model.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import joblib

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def train_model(season: int = 2024):
    """
    Trains an XGBoost regression model on the engineered features.

    Args:
        season (int): The F1 season used for training data.

    Returns:
        model: The trained XGBoost model.
    """

    feature_path = DATA_DIR / f"features_{season}.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing {feature_path}. Run feature_engineering.py first.")

    print(f"🏁 Training model on features from {season} season...")
    df = pd.read_csv(feature_path)

    # Select relevant features
    feature_cols = ["avg_grid", "races", "total_points", "finish_diff", "form_score"]
    target_col = "avg_finish"

    X = df[feature_cols]
    y = df[target_col]

    # Split into training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Initialize XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)
    print("✅ Model training complete!")

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"📊 Evaluation metrics:")
    print(f"   MAE (Mean Absolute Error): {mae:.3f}")
    print(f"   R² score: {r2:.3f}")

    # Save model
    model_path = DATA_DIR / "xgb_model.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

    # Save feature importances
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\n🏎️ Feature importance:")
    print(importances)

    return model


if __name__ == "__main__":
    train_model(2024)
