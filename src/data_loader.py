"""
data_loader.py
---------------
Fetches Formula 1 race results from the Ergast API and saves them as a CSV.
Used as the first step in building your F1 predictor model.
"""

import requests
import pandas as pd
from pathlib import Path


# Create /data folder if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def get_race_results(season: int = 2024) -> pd.DataFrame:
    """
    Fetches all race results for a given Formula 1 season from the Ergast API.

    Args:
        season (int): The F1 season year to fetch data for.

    Returns:
        pd.DataFrame: Cleaned table of race results with key details.
    """
    print(f"Fetching race results for {season} season...")

    # The Ergast API provides paginated JSON
    url = f"https://ergast.com/api/f1/{season}/results.json?limit=1000"
    response = requests.get(url)

    if response.status_code != 200:
        raise ConnectionError(f"Error fetching data: {response.status_code}")

    data = response.json()
    races = data["MRData"]["RaceTable"]["Races"]

    results = []
    for race in races:
        race_name = race["raceName"]
        round_no = int(race["round"])
        date = race["date"]
        circuit = race["Circuit"]["circuitName"]

        for res in race["Results"]:
            driver = res["Driver"]
            constructor = res["Constructor"]

            results.append({
                "season": season,
                "round": round_no,
                "race": race_name,
                "date": date,
                "circuit": circuit,
                "driver": f"{driver['givenName']} {driver['familyName']}",
                "constructor": constructor["name"],
                "grid": int(res["grid"]),
                "position": int(res["position"]),
                "points": float(res["points"]),
                "status": res["status"],
                "fastestLapTime": res.get("FastestLap", {}).get("Time", {}).get("time", None)
            })

    df = pd.DataFrame(results)

    # Save locally
    file_path = DATA_DIR / f"race_results_{season}.csv"
    df.to_csv(file_path, index=False)

    print(f"✅ Saved race results to {file_path}")
    print(f"📊 Loaded {len(df)} race entries across {df['race'].nunique()} races.")
    return df


if __name__ == "__main__":
    # Example usage
    season = 2024
    df = get_race_results(season)
    print(df.head())
