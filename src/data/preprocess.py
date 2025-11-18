#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py
===============================
Data preprocessing script for Air Quality Forecasting Project.

Reads raw data from data/raw/, cleans and standardizes it,
and saves the processed file to data/processed/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


# ============================================================
# 🔧 Load configuration (global.yaml)
# ============================================================
def load_global_config(path="../../config/global.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 🧹 Main preprocessing function
# ============================================================
def preprocess(input_path, output_path, cfg):
    """
    Clean and preprocess the Air Quality dataset.

    Args:
        input_path (str or Path): path to raw CSV file.
        output_path (str or Path): where to save cleaned CSV.
        cfg (dict): global configuration dictionary.
    """

    print(f"📂 Loading raw data from: {input_path}")
    df = pd.read_csv(input_path, sep=";", decimal=",", low_memory=False)

    # Drop unnamed empty columns (some UCI datasets end with ';')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Replace missing values (-200 → NaN)
    missing_value = cfg["data"].get("missing_value", -200)
    df.replace(missing_value, np.nan, inplace=True)

    # Merge Date + Time into DateTime
    if "Date" in df.columns and "Time" in df.columns:
        df["DateTime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H.%M.%S",  # 指定日期格式
            errors="coerce"
        )
        df.drop(columns=["Date", "Time"], inplace=True)

    # Drop rows without DateTime or with all NaNs
    df.dropna(subset=["DateTime"], inplace=True)
    df.dropna(
        how="all",
        subset=[col for col in df.columns if col != "DateTime"],
        inplace=True
    )

    # # Drop specified columns (if configured)
    # drop_cols = cfg["data"].get("drop_columns", [])
    # for col in drop_cols:
    #     if col in df.columns:
    #         df.drop(columns=[col], inplace=True)

    # Sort chronologically & remove duplicates
    df.sort_values(by="DateTime", inplace=True)
    df.drop_duplicates(subset=["DateTime"], keep="first", inplace=True)

    # Optional normalization
    # if cfg["data"].get("normalize", False):
    #     numeric_cols = df.select_dtypes(include=[np.number]).columns
    #     df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    #     print("📏 Numeric columns normalized (z-score).")

    # ============================================================
    # 📉 Time series smoothing (optional)
    # ============================================================
    if cfg["data"].get("smoothing", False):
        window = cfg["data"].get("smoothing_window", 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            df[col] = df[col].rolling(window=window, min_periods=1, center=True).mean()

        print(f"📈 Applied {window}-step moving average smoothing.")

    # ============================================================
    # ⚠️ Outlier detection and handling (optional)
    # ============================================================
    if cfg["data"].get("handle_outliers", False):
        method = cfg["data"].get("outlier_method", "zscore").lower()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "zscore":
            # 以Z-score判断：超过 ±3σ 为异常
            for col in numeric_cols:
                mean, std = df[col].mean(), df[col].std()
                outliers = (df[col] - mean).abs() > 3 * std
                df.loc[outliers, col] = np.nan
            print("⚙️ Outliers handled using Z-score method (|z| > 3).")

        elif method == "iqr":
            # IQR法：超出 [Q1 - 1.5IQR, Q3 + 1.5IQR]
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
                df.loc[outliers, col] = np.nan
            print("⚙️ Outliers handled using IQR method.")

    # Reset index and save
    df.reset_index(drop=True, inplace=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to: {output_path}")
    print(f"🧾 Total records: {len(df)} | Columns: {len(df.columns)}")


# ============================================================
# 🚀 CLI entry point
# ============================================================
if __name__ == "__main__":
    cfg = load_global_config()
    input_path = Path(__file__).resolve().parents[2] / cfg["paths"]["raw_data"]
    output_path = Path(__file__).resolve().parents[2] / cfg["paths"]["processed_data"]
    preprocess(input_path, output_path, cfg)
