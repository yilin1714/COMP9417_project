#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset_classification.py
================================
专门用于 Classification 数据集划分（ONLY hourly version）

输出文件：
    data/splits/X_train_classification.csv
    data/splits/X_val_classification.csv
    data/splits/X_test_classification.csv

以及 horizon 对应的 y 文件：
    y_train_classification_t+0.csv   ← baseline
    y_train_classification_t+1.csv
    y_train_classification_t+6.csv
    y_train_classification_t+12.csv
    y_train_classification_t+24.csv
"""

import pandas as pd
from pathlib import Path
from src.data.preprocess import load_global_config


# =====================================================
# 🔥 Classification Split（ONLY hourly）
# =====================================================
def split_dataset_classification(cfg, val_ratio=0.1):
    root = Path(__file__).resolve().parents[2]
    datetime_col = cfg["data"]["datetime_col"]
    base_target = cfg["classification"]["base_target"]  # "CO(GT)"
    horizons = cfg["classification"]["horizons"]  # [1,6,12,24]

    feature_path = root / "data/features/classification_features.csv"
    print(f"\n📂 Loading classification features: {feature_path}")

    if not feature_path.exists():
        raise FileNotFoundError("❌ classification_features.csv 未找到，请先运行 feature_engineering_classification.py！")

    df = pd.read_csv(feature_path, parse_dates=[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    df = df.dropna(axis=1, how="all")  # 删除空列

    # =====================================================
    # 1️⃣ 按年份划分：2004 → train+val, 2005 → test
    # =====================================================
    train_val_df = df[df[datetime_col].dt.year == 2004]
    test_df = df[df[datetime_col].dt.year == 2005]

    val_start = int(len(train_val_df) * (1 - val_ratio))
    train_df = train_val_df.iloc[:val_start]
    val_df = train_val_df.iloc[val_start:]

    # 缺失值前后填充
    train_df = train_df.ffill().bfill()
    val_df = val_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    # =====================================================
    # 2️⃣ 选择特征列（去掉 label 列）
    # =====================================================
    # 所有 future horizon 的标签列
    class_cols = [f"{base_target}_class_t+{h}" for h in horizons]

    # baseline 的 t+0 标签
    base_col = f"{base_target}_class_t+0"
    class_cols.append(base_col)

    # 其余列都是特征
    feature_cols = [
        c for c in df.columns
        if c not in class_cols + [datetime_col]
    ]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    # =====================================================
    # 3️⃣ 保存所有输出
    # =====================================================
    out_dir = root / "data/splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 X
    X_train.to_csv(out_dir / "X_train_classification.csv", index=False)
    X_val.to_csv(out_dir / "X_val_classification.csv", index=False)
    X_test.to_csv(out_dir / "X_test_classification.csv", index=False)

    # =====================================================
    # 🔵 保存 baseline t+0
    # =====================================================
    if base_col not in df.columns:
        raise ValueError(
            f"❌ 未找到列 {base_col}！请确保 feature_engineering_classification.py 生成了该列。"
        )

    train_df[base_col].to_csv(out_dir / "y_train_classification_t+0.csv", index=False)
    val_df[base_col].to_csv(out_dir / "y_val_classification_t+0.csv", index=False)
    test_df[base_col].to_csv(out_dir / "y_test_classification_t+0.csv", index=False)

    # =====================================================
    # 🔴 保存每个 horizon 的 y
    # =====================================================
    for h in horizons:
        col_name = f"{base_target}_class_t+{h}"

        train_df[col_name].to_csv(out_dir / f"y_train_classification_t+{h}.csv", index=False)
        val_df[col_name].to_csv(out_dir / f"y_val_classification_t+{h}.csv", index=False)
        test_df[col_name].to_csv(out_dir / f"y_test_classification_t+{h}.csv", index=False)

    print("✅ Classification dataset split complete!")
    return X_train, X_val, X_test


# =====================================================
# 🚀 MAIN ENTRY
# =====================================================
if __name__ == "__main__":
    cfg = load_global_config()
    split_dataset_classification(cfg)
    print("\n🎉 All classification splits completed!")
