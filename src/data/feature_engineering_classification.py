#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_engineering_classification.py
=====================================
Create classification features according to project requirements.

Classification target: CO(GT)
Classes:
    low  < 1.5
    mid  1.5 ≤ CO < 2.5
    high > 2.5

Forecast horizons:
    t+0, t+1, t+6, t+12, t+24
(ONLY hourly classification needed — daily removed)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocess import load_global_config


# ============================================================
# 🟦 Discretise CO(GT) into 3 classes
# ============================================================
def discretise_co(value):
    if value < 1.5:
        return "low"
    elif value < 2.5:
        return "mid"
    else:
        return "high"


# ============================================================
# 🟩 Add engineered features
# ============================================================
def add_features(df, cfg):
    datetime_col = cfg["data"]["datetime_col"]
    target = cfg["classification"]["base_target"]
    fe = cfg["data"]["feature_engineering"]

    lookback = fe["lookback"]
    roll_windows = fe["roll_windows"]
    use_time = fe["include_time_features"]
    use_cyclic = fe["use_cyclical_encoding"]

    # ---- Lag features ----
    for lag in range(1, lookback + 1):
        df[f"{target}_t-{lag}h"] = df[target].shift(lag)

    # ---- Rolling windows ----
    for w in roll_windows:
        df[f"{target}_roll_mean_{w}h"] = df[target].rolling(w).mean()
        df[f"{target}_roll_std_{w}h"] = df[target].rolling(w).std()

    # ---- Time features ----
    if use_time:
        df["weekday"] = df[datetime_col].dt.dayofweek
        df["month"] = df[datetime_col].dt.month
        df["hour"] = df[datetime_col].dt.hour

        if use_cyclic:
            df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


# ============================================================
# 🟥 Add classification labels for all horizons (including t+0 baseline)
# ============================================================
def add_class_labels(df, cfg):
    target = cfg["classification"]["base_target"]      # "CO(GT)"
    horizons = cfg["classification"]["horizons"]       # [1,6,12,24]

    # 🔵 baseline t+0
    df[f"{target}_class_t+0"] = df[target].apply(discretise_co)

    # 🔴 future horizons: t+1, t+6, t+12, t+24
    for h in horizons:
        df[f"{target}_t+{h}h"] = df[target].shift(-h)
        df[f"{target}_class_t+{h}"] = df[f"{target}_t+{h}h"].apply(discretise_co)

    return df


# ============================================================
# ⭐ MAIN
# ============================================================
def create_classification_features(input_path, output_path, cfg):
    print("\n📂 Creating HOURLY CLASSIFICATION FEATURES...")

    datetime_col = cfg["data"]["datetime_col"]

    # Load processed data
    df = pd.read_csv(input_path, parse_dates=[datetime_col]).sort_values(datetime_col)

    # Drop unwanted columns
    drop_cols = cfg["data"].get("drop_columns", [])
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ----- Add labels -----
    df = add_class_labels(df, cfg)

    # ----- Add features -----
    df = add_features(df, cfg)

    # ----- Remove rows where shifted labels become NaN -----
    future_label_cols = [
        f"{cfg['classification']['base_target']}_class_t+{h}"
        for h in cfg["classification"]["horizons"]
    ]
    df = df.dropna(subset=future_label_cols).reset_index(drop=True)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Saved classification features → {output_path}")
    print(f"📊 Shape: {df.shape}")


# ============================================================
# 🚀 ENTRY POINT
# ============================================================
if __name__ == "__main__":
    cfg = load_global_config()
    root = Path(__file__).resolve().parents[2]
    input_path = root / cfg["paths"]["processed_data"]
    output_path = root / "data/features/classification_features.csv"

    create_classification_features(input_path, output_path, cfg)

    print("\n🎉 Classification feature engineering complete!")
