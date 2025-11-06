# 🏎️ Formula 1 Race Predictor  

### A data-driven machine learning model that predicts Formula 1 race results and season standings using real-world data, driver performance metrics, and Monte Carlo simulations.  

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-yellow.svg)
![Status](https://img.shields.io/badge/Project%20Status-Complete-blue.svg)


---

## 🚀 Overview

This project builds an end-to-end **Formula 1 race prediction system** that:
- Fetches historical race results from the **Ergast API** (or uses sample offline data)
- Engineers driver and constructor performance features
- Trains an **XGBoost regression model** to predict finishing positions
- Runs **Monte Carlo simulations** to estimate World Drivers’ Championship standings  

All components are modular and reproducible in **GitHub Codespaces** or locally.

---

## 🧠 Features

- 📥 Fetch and store historical F1 race results  
- 🧮 Engineer performance features (average grid, consistency, win rate, etc.)  
- 🤖 Train an ML model to predict finishing positions  
- 🎲 Simulate full seasons thousands of times for probabilistic standings  
- 📊 Visualize predictions and championship outcomes in notebooks  

---

## 📂 Repository Structure

```text
f1-predictor/
│
├── src/                                # Source code for the F1 predictor
│   ├── __init__.py                     # Package initializer
│   ├── data_loader.py                  # Fetches and stores race results
│   ├── feature_engineering.py          # Builds driver/team performance features
│   ├── model_training.py               # Trains XGBoost model on engineered data
│   ├── simulation.py                   # Monte Carlo simulation for championship standings
│   └── utils.py                        # Utility functions (I/O, data checks, helpers)
│
├── data/                               # Local datasets and model outputs
│   ├── race_results_2024.csv           # Sample raw race data (API or offline)
│   ├── features_2024.csv               # Engineered features for training
│   ├── simulated_standings_2025.csv    # Simulated driver standings
│   └── xgb_model.joblib                # Trained model file (generated automatically)
│
├── notebooks/                          # Jupyter notebooks for exploration and visualization
│   └── exploration.ipynb               # Interactive analysis and visualizations
│
├── tests/                              # Automated tests to ensure functionality
│   └── test_model.py                   # Unit tests for feature building and model training
│
├── .gitignore                          # Git ignore rules (keeps repo clean)
├── LICENSE                             # Apache 2.0 license
├── README.md                           # Project documentation (you’re reading it!)
├── requirements.txt                    # Python dependencies
└── NOTICE                              # Optional credits and notices

```
## ⚙️ Installation

### Option 1 — Run in **GitHub Codespaces** (recommended)
1. Click **“Code → Open with Codespaces”**
2. Once the environment loads, install dependencies:
   ```bash
   pip install -r requirements.txt

Option 2 — Run locally

1. Clone the repo:

git clone https://github.com/MRuhan17/f1-predictor.git
cd f1-predictor


2. Create a virtual environment:

python -m venv venv
source venv/bin/activate   


3. Install dependencies:

pip install -r requirements.txtInstall dependencies:

pip install -r requirements.txt
Usage

Run the pipeline step-by-step:

# 1️⃣ Load or generate race results
python src/data_loader.py

# 2️⃣ Build engineered features
python src/feature_engineering.py

# 3️⃣ Train the prediction model
python src/model_training.py

# 4️⃣ Run Monte Carlo season simulation
python src/simulation.py


Then explore your results visually in the notebook:

jupyter notebook notebooks/exploration.ipynb


🧩 Example outputs

Engineered features: data/features_2024.csv

Trained model: data/xgb_model.joblib

Simulated standings: data/simulated_standings_2025.csv

Driver	Avg Points	Rank
Max Verstappen	24.8	1
Lando Norris	23.9	2
Charles Leclerc	22.7	3

Testing

Run automated tests using pytest:

pytest -v


Future enhancements

Integrate FastF1 for live telemetry & lap data

Add Streamlit dashboard for interactive race prediction

Support Constructor Championship simulations

Introduce track-specific performance modifiers

📜 License

This project is licensed under the Apache License 2.0


👨‍💻 Author
Ruhulalemeen Mulla
Machine Learning Enthusiast | F1 Analytics Developer


📧 ruhanmulla07@gmail.com

