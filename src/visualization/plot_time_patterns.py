#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_time_patterns.py
=========================
📈 Visualize temporal patterns (daily, monthly, weekly) of pollutants.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.preprocess import load_global_config


def plot_time_patterns(cfg):
    """
    依据 cfg 配置文件可视化时间规律（小时/月/星期平均污染水平）
    """
    # === 1️⃣ 路径构造 ===
    root_dir = Path(__file__).resolve().parents[2]
    data_path = root_dir / cfg["paths"]["processed_data"]
    save_dir = root_dir / cfg["paths"]["plots"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # === 2️⃣ 数据读取 ===
    df = pd.read_csv(data_path)
    datetime_col = cfg["data"]["datetime_col"]
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # 自动检测污染物列
    pollutants = cfg["data"].get("all_pollutants") or [
        col for col in df.columns if col not in [datetime_col]
    ]

    # === 🧩 2.1 缺失值处理 ===
    # 先按时间排序，防止时间乱序导致插值错误
    df = df.sort_values(by=datetime_col)

    # 对污染物列线性插值 + 前后填充兜底
    df[pollutants] = (
        df[pollutants]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    # === 3️⃣ 添加时间特征 ===
    df["Hour"] = df[datetime_col].dt.hour
    df["Month"] = df[datetime_col].dt.month
    df["Weekday"] = df[datetime_col].dt.day_name()

    # === 4️⃣ 每小时平均趋势 ===
    hourly_mean = df.groupby("Hour")[pollutants].mean()
    plt.figure(figsize=(10, 6))
    hourly_mean.plot(ax=plt.gca(), linewidth=1.5)
    plt.title("Average Hourly Pollution Levels")
    plt.xlabel("Hour of Day")
    plt.ylabel("Concentration")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "timepattern_hourly.png", dpi=300)
    plt.close()

    # === 5️⃣ 每月平均趋势 ===
    monthly_mean = df.groupby("Month")[pollutants].mean()
    plt.figure(figsize=(10, 6))
    monthly_mean.plot(ax=plt.gca(), linewidth=1.5)
    plt.title("Average Monthly Pollution Levels")
    plt.xlabel("Month")
    plt.ylabel("Concentration")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "timepattern_monthly.png", dpi=300)
    plt.close()

    # === 6️⃣ 每周（星期）平均趋势 ===
    weekday_mean = (
        df.groupby("Weekday")[pollutants].mean()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    )
    plt.figure(figsize=(10, 6))
    weekday_mean.plot(ax=plt.gca(), linewidth=1.5)
    plt.title("Average Weekly Pollution Levels")
    plt.xlabel("Day of Week")
    plt.ylabel("Concentration")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "timepattern_weekly.png", dpi=300)
    plt.close()

    print(f"✅ Saved all time pattern plots to: {save_dir.resolve()}")


if __name__ == "__main__":
    cfg = load_global_config()
    plot_time_patterns(cfg)
