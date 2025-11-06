"""
feature_engineering.py
----------------------
Processes raw race results into structured driver and constructor features.
Creates performance metrics such as average grid, finish, consistency,
and relative improvement across races.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def build_features(season: int = 2024) -> pd.DataFrame:
    """
    Builds a feature table for a given season based on raw race results.

    Args:
        season (int): The F1 season to process.

    Returns:
        pd.DataFrame: Driver-level performance features.
    """

    raw_path = DATA_DIR / f"race_results_{season}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run data_loader.py first.")

    print(f"🔧 Building features for {season} season...")
    df = pd.read_csv(raw_path)

    # Convert positions and grids to numeric
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")

    # Basic driver-level stats
    driver_stats = (
        df.groupby(["driver", "constructor"])
          .agg(
              avg_grid=("grid", "mean"),
              avg_finish=("position", "mean"),
              races=("race", "count"),
              total_points=("points", "sum"),
              wins=("position", lambda x: (x == 1).sum()),
              dnfs=("status", lambda x: (x != "Finished").sum())
          )
          .reset_index()
    )

    # Derived performance metrics
    driver_stats["finish_diff"] = driver_stats["avg_grid"] - driver_stats["avg_finish"]
    driver_stats["consistency"] = 1 / (1 + driver_stats["avg_finish"].std() if driver_stats["races"].all() else 1)
    driver_stats["win_rate"] = driver_stats["wins"] / driver_stats["races"]
    driver_stats["dnf_rate"] = driver_stats["dnfs"] / driver_stats["races"]
    driver_stats["form_score"] = (
        0.5 * (25 * driver_stats["win_rate"])  # wins are huge
        + 0.3 * (driver_stats["finish_diff"])
        + 0.2 * (driver_stats["total_points"] / driver_stats["races"])
        - 10 * driver_stats["dnf_rate"]
    )

    # Normalize form_score to 0–100 scale
    driver_stats["form_score"] = (
        100 * (driver_stats["form_score"] - driver_stats["form_score"].min())
        / (driver_stats["form_score"].max() - driver_stats["form_score"].min())
    )

    # Sort by performance
    driver_stats.sort_values("form_score", ascending=False, inplace=True)

    # Save features
    feature_path = DATA_DIR / f"features_{season}.csv"
    driver_stats.to_csv(feature_path, index=False)
    print(f"✅ Saved engineered features to {feature_path}")
    print(driver_stats.head(10))

    return driver_stats


if __name__ == "__main__":
    build_features(2024)
