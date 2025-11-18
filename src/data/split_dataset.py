#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py
========================
Split the feature-enhanced Air Quality dataset
into train / validation / test sets based on time.

Now supports directly passing in the feature file name.
"""

import pandas as pd
from pathlib import Path
from src.data.preprocess import load_global_config
import argparse


def split_dataset(features_file, cfg, val_ratio=0.1):
    root = Path(__file__).resolve().parents[2]
    datetime_col = cfg["data"]["datetime_col"]
    targets = cfg["data"]["targets"]

    # 选择 hourly 或 daily 的前缀
    prefix = features_file.replace("_features.csv", "").replace(".csv", "")

    # 1️⃣ Load feature file
    feature_path = root / "data/features" / features_file
    print(f"📂 Loading features from: {feature_path}")

    df = pd.read_csv(feature_path, parse_dates=[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # 删除全 NaN 列
    df = df.dropna(axis=1, how="all")

    # 2️⃣ Split
    train_val_df = df[df[datetime_col].dt.year == 2004]
    test_df = df[df[datetime_col].dt.year == 2005]

    val_idx = int(len(train_val_df) * (1 - val_ratio))
    train_df = train_val_df.iloc[:val_idx]
    val_df = train_val_df.iloc[val_idx:]

    # Fill NaN
    train_df = train_df.ffill().bfill()
    val_df = val_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    feature_cols = [c for c in df.columns if c not in targets + [datetime_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[targets]

    X_val = val_df[feature_cols]
    y_val = val_df[targets]

    X_test = test_df[feature_cols]
    y_test = test_df[targets]

    # 3️⃣ Save split files — with prefix!
    output_dir = root / "data" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / f"X_train_{prefix}.csv", index=False)
    y_train.to_csv(output_dir / f"y_train_{prefix}.csv", index=False)
    X_val.to_csv(output_dir / f"X_val_{prefix}.csv", index=False)
    y_val.to_csv(output_dir / f"y_val_{prefix}.csv", index=False)
    X_test.to_csv(output_dir / f"X_test_{prefix}.csv", index=False)
    y_test.to_csv(output_dir / f"y_test_{prefix}.csv", index=False)

    print(f"💾 Saved split files for {prefix} in data/splits/")
    print("✅ Dataset split completed.")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Feature file name (e.g., hourly_features.csv)")
    args = parser.parse_args()

    cfg = load_global_config()
    split_dataset(args.features, cfg)
