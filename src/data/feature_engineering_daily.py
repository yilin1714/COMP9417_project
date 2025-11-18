#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_engineering_daily.py
============================
Generate DAILY feature set only.
Completely independent version (no shared utils).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocess import load_global_config


# ============================================================
# 🧩 Feature Engineering (daily version)
# ============================================================
def create_features_daily(input_path, output_path, cfg):
    print(f"📂 Loading cleaned data from: {input_path}")

    df = pd.read_csv(input_path, parse_dates=[cfg["data"]["datetime_col"]])
    df.sort_values(cfg["data"]["datetime_col"], inplace=True)

    datetime_col = cfg["data"]["datetime_col"]
    targets = cfg["data"]["targets"]
    fe_params = cfg["data"]["feature_engineering"]

    # 强制 daily 模式
    granularity = "daily"
    horizons = cfg["prediction"]["horizons_daily"]

    lookback = fe_params.get("lookback", 3)
    roll_windows = fe_params.get("roll_windows", [3, 6, 12])
    include_time_features = fe_params.get("include_time_features", True)
    use_cyclical_encoding = fe_params.get("use_cyclical_encoding", True)

    # =======================================================
    # ⭐ 0️⃣ Resample HOURLY → DAILY
    # =======================================================
    print("📆 Converting hourly data → DAILY resolution...")

    df = df.set_index(datetime_col)

    base_agg = cfg["data"].get("daily", {}).get("agg", {})
    agg_dict = {col: base_agg.get(col, "mean") for col in df.columns}

    df = df.resample("D").agg(agg_dict)
    df = df.reset_index()

    print(f"📊 After DAILY resample shape: {df.shape}")

    # =======================================================
    # Drop columns
    # =======================================================
    drop_cols = cfg["data"].get("drop_columns", [])
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # =======================================================
    # 1️⃣ Lag Features
    # =======================================================
    print(f"🕒 Generating DAILY lag features (lookback={lookback})...")
    lag_features = {}
    for lag in range(1, lookback + 1):
        for tgt in targets:
            lag_features[f"{tgt}_t-{lag}"] = df[tgt].shift(lag)

    # =======================================================
    # 2️⃣ Rolling Features (daily)
    # =======================================================
    print("📈 Generating DAILY rolling features...")
    roll_features = {}
    suffix = "d"

    for tgt in targets:
        for w in roll_windows:
            roll_features[f"{tgt}_roll_mean_{w}{suffix}"] = df[tgt].rolling(window=w).mean()
            roll_features[f"{tgt}_roll_std_{w}{suffix}"] = df[tgt].rolling(window=w).std()

    df = pd.concat([df, pd.DataFrame(lag_features), pd.DataFrame(roll_features)], axis=1)

    # =======================================================
    # 3️⃣ Time Features (NO hour)
    # =======================================================
    if include_time_features:
        print("🗓️ Adding DAILY time features...")

        df["weekday"] = df[datetime_col].dt.dayofweek
        df["month"] = df[datetime_col].dt.month

        # daily → 没有 hour，设置为空
        df["hour"] = np.nan

        if use_cyclical_encoding:
            df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            # 不生成 hour_sin / hour_cos（daily 不需要）
            df["hour_sin"] = np.nan
            df["hour_cos"] = np.nan

    # =======================================================
    # 4️⃣ Multi-horizon Targets (daily)
    # =======================================================
    print(f"⏩ Generating DAILY future targets: {horizons}")
    for tgt in targets:
        for h in horizons:
            df[f"{tgt}_t+{h}{suffix}"] = df[tgt].shift(-h)


    # =======================================================
    # 5️⃣ Save
    # =======================================================
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ DAILY feature file saved to: {output_path}")
    print(f"📊 Final DAILY shape: {df.shape}")


# ============================================================
# 🚀 Script Entry
# ============================================================
if __name__ == "__main__":
    cfg = load_global_config()
    root = Path(__file__).resolve().parents[2]

    input_path = root / cfg["paths"]["processed_data"]

    # 保存到 data/features/daily/daily_features.csv
    features_dir = root / cfg["paths"]["features_dir"]
    features_dir.mkdir(parents=True, exist_ok=True)
    output_path = features_dir / "daily_features.csv"

    create_features_daily(input_path, output_path, cfg)

    print("\n🎉 DAILY feature generation completed!\n")
