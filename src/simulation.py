"""
simulation.py
--------------
Uses the trained XGBoost model to simulate Formula 1 race results
and estimate end-of-season standings using Monte Carlo simulation.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# F1 points system for top 10 finishers
POINTS_TABLE = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def simulate_race(model, features_df, randomness: float = 0.15) -> pd.DataFrame:
    """
    Simulates a single race using model predictions with noise added.

    Args:
        model: Trained XGBoost model.
        features_df (pd.DataFrame): Driver feature data.
        randomness (float): Amount of random variation in finishing position (0–1).

    Returns:
        pd.DataFrame: Predicted race results (sorted by predicted finish).
    """
    preds = model.predict(features_df[["avg_grid", "races", "total_points", "finish_diff", "form_score"]])
    results = features_df.copy()
    results["predicted_finish"] = preds + np.random.normal(0, randomness, size=len(preds))
    results.sort_values("predicted_finish", inplace=True)
    results["predicted_position"] = np.arange(1, len(results) + 1)
    return results


def simulate_season(season: int = 2025, n_races: int = 22, n_runs: int = 1000):
    """
    Runs Monte Carlo simulations of an F1 season and estimates driver standings.

    Args:
        season (int): The season year (for file names).
        n_races (int): Number of races in the simulated season.
        n_runs (int): Number of Monte Carlo simulations.

    Returns:
        pd.DataFrame: Average predicted championship standings.
    """
    model_path = DATA_DIR / "xgb_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run model_training.py first.")

    model = joblib.load(model_path)

    # Use the most recent feature set as a proxy for next season
    features_path = DATA_DIR / "features_2024.csv"
    if not features_path.exists():
        raise FileNotFoundError("Feature file not found. Run feature_engineering.py first.")

    features_df = pd.read_csv(features_path)
    drivers = features_df["driver"].tolist()
    championship_points = {driver: 0 for driver in drivers}

    print(f"🏎️ Simulating {n_runs} seasons with {n_races} races each...")

    for run in range(n_runs):
        for _ in range(n_races):
            race_results = simulate_race(model, features_df)
            # Assign points based on predicted finish
            for idx, row in race_results.iterrows():
                if row["predicted_position"] <= len(POINTS_TABLE):
                    pts = POINTS_TABLE[int(row["predicted_position"]) - 1]
                    championship_points[row["driver"]] += pts

        # Small random variation to simulate season unpredictability
        for driver in drivers:
            championship_points[driver] += np.random.normal(0, 5)

    # Convert to standings
    standings = pd.DataFrame.from_dict(championship_points, orient="index", columns=["total_points"])
    standings.sort_values("total_points", ascending=False, inplace=True)
    standings.reset_index(inplace=True)
    standings.rename(columns={"index": "driver"}, inplace=True)

    # Normalize across runs
    standings["avg_points"] = standings["total_points"] / n_runs

    # Save and display
    output_path = DATA_DIR / f"simulated_standings_{season}.csv"
    standings.to_csv(output_path, index=False)
    print(f"✅ Simulation complete! Saved to {output_path}")
    print(standings.head(10))

    return standings


if __name__ == "__main__":
    simulate_season()
