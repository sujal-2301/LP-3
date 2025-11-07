#!/usr/bin/env python3
"""
Uber Trips — College Practical Script (EDA + Cleaning + Simple Model)
Author: <your name>
Run:
    pip install pandas numpy matplotlib scikit-learn
    python uber_practical.py --csv uber.csv

What it does:
1) Loads CSV and prints dataset summary.
2) Cleans obvious issues (nulls, invalid coords, bad fares/passenger counts).
3) Engineers features (hour, day_of_week, trip distance via Haversine).
4) Draws basic plots (saved to ./plots).
5) Trains a simple Linear Regression model: fare ~ distance_km + hour + passenger_count.
6) Reports R^2 on train/test and prints coefficients.
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------
# Utility
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance in KM.
    Inputs/outputs are numpy arrays or pandas Series.
    """
    R = 6371.0  # Earth radius (km)
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def plot_and_save(series_or_df, kind, title, xlabel, ylabel, out_dir: Path, filename: str):
    """One chart per figure, Matplotlib only, no color specified."""
    plt.figure()
    if kind == "bar":
        series_or_df.plot(kind="bar")
    elif kind == "line":
        series_or_df.plot(kind="line")
    elif kind == "hist":
        series_or_df.plot(kind="hist", bins=50)
    elif kind == "scatter":
        # expecting a DataFrame with x and y columns already selected
        plt.scatter(series_or_df.iloc[:,0], series_or_df.iloc[:,1], s=5)
    else:
        raise ValueError("Unsupported plot kind")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    ensure_dir(out_dir)
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150)
    try:
        plt.show()
    except Exception:
        # Non-interactive environments may not support show()
        pass
    plt.close()
    return out_path

# ----------------------------
# Core Workflow
# ----------------------------
def load_dataset(csv_path: Path, sample: int = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
    # Drop unnamed index-like columns
    for c in list(df.columns):
        if c.lower().startswith("unnamed"):
            df = df.drop(columns=[c])
    return df.reset_index(drop=True)

def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "pickup_datetime", "date_time", "datetime", "timestamp",
        "date/time", "date", "time"
    ]
    dt_col = None
    for cand in candidates:
        for c in df.columns:
            if c.lower() == cand.lower():
                dt_col = c; break
        if dt_col: break
    if dt_col is None:
        # fuzzy
        for c in df.columns:
            cl = c.lower()
            if "pickup" in cl and "time" in cl:
                dt_col = c; break
    if dt_col is None:
        raise ValueError("No datetime-like column found (e.g., 'pickup_datetime').")
    df["pickup_datetime"] = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True)
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.day_name()
    df["date"] = df["pickup_datetime"].dt.date
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only essential columns if present
    required = ["fare_amount","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count","pickup_datetime"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}")
    # Remove rows with nulls in essential columns
    df = df.dropna(subset=required)
    # NYC-ish coordinate sanity filter (also guards against swapped lon/lat)
    lat_ok = df["pickup_latitude"].between(40.0, 42.0) & df["dropoff_latitude"].between(40.0, 42.0)
    lon_ok = df["pickup_longitude"].between(-75.0, -72.0) & df["dropoff_longitude"].between(-75.0, -72.0)
    df = df[lat_ok & lon_ok]
    # Passenger count: keep 1..6
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]
    # Fare must be positive and reasonable
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 300)]
    # Drop duplicates
    df = df.drop_duplicates()
    return df.reset_index(drop=True)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["distance_km"] = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    # remove zero/insane distances
    df = df[(df["distance_km"] > 0.05) & (df["distance_km"] < 100)].copy()
    return df

def eda_plots(df: pd.DataFrame, out_dir: Path):
    # Trips by hour
    hour_counts = df["hour"].value_counts().sort_index()
    plot_and_save(hour_counts, "bar", "Trips by Hour of Day", "Hour", "Count", out_dir, "trips_by_hour.png")
    # Trips by day of week (ensure order Mon..Sun)
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_counts = df["day_of_week"].value_counts().reindex(order)
    plot_and_save(dow_counts, "bar", "Trips by Day of Week", "Day", "Count", out_dir, "trips_by_dow.png")
    # Daily trip counts
    daily = df.groupby("date").size()
    plot_and_save(daily, "line", "Daily Trip Counts", "Date", "Trips", out_dir, "daily_trips.png")
    # Fare histogram
    plot_and_save(df["fare_amount"], "hist", "Fare Amount Distribution", "Fare ($)", "Frequency", out_dir, "fare_hist.png")
    # Distance histogram
    plot_and_save(df["distance_km"], "hist", "Trip Distance Distribution", "Distance (km)", "Frequency", out_dir, "distance_hist.png")
    # Fare vs Distance scatter (sample to keep light)
    small = df.sample(n=min(20000, len(df)), random_state=42)[["distance_km","fare_amount"]]
    plot_and_save(small, "scatter", "Fare vs Distance (sample)", "Distance (km)", "Fare ($)", out_dir, "fare_vs_distance_scatter.png")

def simple_model(df: pd.DataFrame):
    feats = ["distance_km","hour","passenger_count"]
    X = df[feats].astype(float).values
    y = df["fare_amount"].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    metrics = {
        "train_r2": float(r2_score(y_train, y_pred_tr)),
        "test_r2": float(r2_score(y_test, y_pred_te)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_te)),
        "coef_distance_km": float(model.coef_[0]),
        "coef_hour": float(model.coef_[1]),
        "coef_passenger_count": float(model.coef_[2]),
        "intercept": float(model.intercept_),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }
    return model, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to uber.csv")
    ap.add_argument("--sample", type=int, default=120000, help="Optional sample size to speed up (default 120k)")
    ap.add_argument("--plots_dir", type=str, default="plots", help="Directory to save plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    plots_dir = Path(args.plots_dir)
    ensure_dir(plots_dir)

    print(">>> Loading dataset...")
    df = load_dataset(csv_path, sample=args.sample)
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))

    print("\n>>> Parsing datetime...")
    df = parse_datetime_columns(df)
    print("Parsed 'pickup_datetime' and derived ['hour', 'day_of_week', 'date'].")

    print("\n>>> Basic cleaning...")
    df = basic_cleaning(df)
    print(f"After cleaning -> Rows: {len(df)}")

    print("\n>>> Feature engineering...")
    df = engineer_features(df)
    print("Added 'distance_km'.")

    print("\n>>> Saving quick summaries...")
    summary = df.describe(include="all", datetime_is_numeric=True)
    summary_path = Path("summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path.resolve()}")

    print("\n>>> Making plots...")
    eda_plots(df, plots_dir)
    print(f"Saved plots to {plots_dir.resolve()}")

    print("\n>>> Training simple Linear Regression...")
    _, metrics = simple_model(df)
    for k,v in metrics.items():
        print(f"{k}: {v}")

    # Save cleaned & features file for reference
    out_csv = Path("uber_clean_features.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote cleaned dataset with features to: {out_csv.resolve()}")
    print("\nDone.")

if __name__ == "__main__":
    main()
