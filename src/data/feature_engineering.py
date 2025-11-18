#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_engineering.py
========================
Generate time-series features & multi-horizon targets
from cleaned Air Quality dataset.

Config-driven version:
 - Supports multiple target pollutants
 - Generates lag, rolling, and cyclical time features
 - Creates future prediction targets (t+1, t+6, t+12, t+24)
 - Keeps NaN (handled later in split_dataset)
 - Saves final DataFrame to data/features/features_train.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocess import load_global_config


# ============================================================
# 🧩 Feature Engineering Function
# ============================================================
def create_features(input_path, output_path, cfg):
    """
    Create time-series features and multi-horizon targets.

    Args:
        input_path (str): Path to cleaned CSV file.
        output_path (str): Path to save the feature CSV.
        cfg (dict): Loaded configuration dictionary from YAML.
    """
    print(f"📂 Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=[cfg["data"]["datetime_col"]])
    df.sort_values(cfg["data"]["datetime_col"], inplace=True)

    datetime_col = cfg["data"]["datetime_col"]
    targets = cfg["data"]["targets"]
    fe_params = cfg["data"]["feature_engineering"]
    horizons = cfg.get("prediction", {}).get("horizons", [1, 6, 12, 24])

    lookback = fe_params.get("lookback", 3)
    roll_windows = fe_params.get("roll_windows", [3, 6, 12])
    include_time_features = fe_params.get("include_time_features", True)
    use_cyclical_encoding = fe_params.get("use_cyclical_encoding", True)

    # Drop specified columns (if configured)
    drop_cols = cfg["data"].get("drop_columns", [])
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # =======================================================
    # 1️⃣ 滞后特征（Lag Features）
    # =======================================================
    print(f"🕒 Generating lag features (lookback={lookback}) for all variables...")
    lag_features = {}
    for lag in range(1, lookback + 1):
        for tgt in targets:
            lag_features[f"{tgt}_t-{lag}"] = df[tgt].shift(lag)

    # =======================================================
    # 2️⃣ 滚动统计特征（Rolling Mean/Std）
    # =======================================================
    print(f"📈 Generating rolling mean/std features for targets: {targets}")
    roll_features = {}
    for tgt in targets:
        for w in roll_windows:
            roll_features[f"{tgt}_roll_mean_{w}h"] = df[tgt].rolling(window=w).mean()
            roll_features[f"{tgt}_roll_std_{w}h"] = df[tgt].rolling(window=w).std()

    # 一次性拼接新特征
    df = pd.concat([df, pd.DataFrame(lag_features), pd.DataFrame(roll_features)], axis=1)
    df = df.copy()  # defragmentation

    # =======================================================
    # 3️⃣ 时间特征（hour / weekday / month + cyclical encoding）
    # =======================================================
    if include_time_features:
        print("🗓️ Adding time-based features...")
        df["hour"] = df[datetime_col].dt.hour
        df["weekday"] = df[datetime_col].dt.dayofweek
        df["month"] = df[datetime_col].dt.month

        if use_cyclical_encoding:
            print("🔁 Applying cyclical encoding for time features...")
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # =======================================================
    # 4️⃣ 创建未来预测目标（Multi-horizon Targets）
    # =======================================================
    print(f"⏩ Generating future targets for horizons: {horizons} hours")
    for tgt in targets:
        for h in horizons:
            df[f"{tgt}_t+{h}"] = df[tgt].shift(-h)

    # =======================================================
    # 5️⃣ 保留 NaN（后续由 split_dataset 处理）
    # =======================================================
    nan_count = df.isna().sum().sum()
    print(f"⚠️ Keeping NaN rows ({nan_count:,} NaN values exist after feature creation).")

    # =======================================================
    # 6️⃣ 保存结果
    # =======================================================
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Feature file saved to: {output_path}")
    print(f"📊 Final shape: {df.shape}")


# ============================================================
# 🚀 Main Entry
# ============================================================
if __name__ == "__main__":
    cfg = load_global_config()
    input_path = Path(__file__).resolve().parents[2] / cfg["paths"]["processed_data"]
    output_path = Path(__file__).resolve().parents[2] / cfg["paths"]["features_data"]

    print("\n🚀 Starting Feature Engineering Pipeline (multi-horizon)...")
    create_features(input_path, output_path, cfg)
    print("✅ All features & targets generated successfully!")
