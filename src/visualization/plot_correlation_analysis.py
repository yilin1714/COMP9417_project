#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correlation_analysis.py
===========================
📊 Compute and visualize correlations among pollutants.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.preprocess import load_global_config


def analyze_pollutant_correlation(data_path, datetime_col='Date', pollutants=None, save_path=None):
    """
    计算并可视化污染物之间的相关性。
    Args:
        data_path (str): CSV文件路径
        datetime_col (str): 日期列名
        pollutants (list[str]): 要分析的污染物列名（默认检测所有数值列）
        save_path (str): 保存热力图路径（可选）
    """
    # 1️⃣ 读取数据
    df = pd.read_csv(data_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # 2️⃣ 自动检测数值列
    if pollutants is None:
        pollutants = [col for col in df.columns if col not in [datetime_col]]

    # 3️⃣ 按天平均 + 插值平滑
    df_daily = (
        df.set_index(datetime_col)
          .resample('D')[pollutants]
          .mean()
          .interpolate(method='linear')
          .ffill()
          .bfill()
    )

    # 4️⃣ 计算相关矩阵（默认 Pearson）
    corr_matrix = df_daily.corr(method='pearson')

    print("📈 Pearson Correlation Matrix:")
    print(corr_matrix.round(3))

    # 5️⃣ 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Between Pollutants (Daily Average)", fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved correlation heatmap to {save_path}")
    else:
        plt.show()

    return corr_matrix


if __name__ == "__main__":
    cfg = load_global_config()
    data_path = Path(__file__).resolve().parents[2] / cfg["paths"]["processed_data"]
    analyze_pollutant_correlation(
        data_path=data_path,
        datetime_col=cfg["data"]["datetime_col"],
        pollutants=cfg["data"]["all_pollutants"],
        save_path=Path(__file__).resolve().parents[2] / cfg["paths"]["plots"] / "pollutant_correlation_heatmap.png"
    )
