"""
f1-predictor
------------
A modular machine learning package that predicts Formula 1 race outcomes
and season standings using historical data and Monte Carlo simulations.

Modules:
- data_loader: Fetches raw race results from the Ergast API
- feature_engineering: Builds driver/team performance features
- model_training: Trains XGBoost model to predict finishing performance
- simulation: Runs Monte Carlo simulations for championship outcomes
- utils: Helper functions for loading, saving, and pipeline checks
"""

__version__ = "0.1.0"
__author__ = "YOUR NAME"

# Re-export commonly used functions for easy importing
from .data_loader import get_race_results
from .feature_engineering import build_features
from .model_training import train_model
from .simulation import simulate_season
from .utils import load_csv, save_csv, preview, check_data_pipeline

__all__ = [
    "get_race_results",
    "build_features",
    "train_model",
    "simulate_season",
    "load_csv",
    "save_csv",
    "preview",
    "check_data_pipeline",
]
