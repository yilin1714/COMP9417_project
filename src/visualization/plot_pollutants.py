#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pollutants.py
==========================
📊 Visualize pollutant concentration time series.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.preprocess import load_global_config

def plot_pollutant_timeseries(data_path, datetime_col='Date', pollutants=None, save_path=None):
    """
    绘制多个污染物的每日平均浓度折线图（自动处理缺失值）。
    Args:
        data_path (str or Path): 数据文件路径 (CSV)
        datetime_col (str): 日期时间列名
        pollutants (list[str]): 要绘制的污染物列
        save_path (str or Path): 保存路径（可选）
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # 1️⃣ 读取数据
    df = pd.read_csv(data_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # 2️⃣ 自动检测数值列
    if pollutants is None:
        pollutants = [col for col in df.columns if col not in [datetime_col]]

    # 3️⃣ 按天求平均
    df_daily = (
        df.set_index(datetime_col)
          .resample('D')[pollutants]
          .mean()
          .reset_index()
    )

    # 4️⃣ 缺失值处理（线性插值 + 前向填充兜底）
    df_daily = df_daily.interpolate(method='linear').ffill()

    # 5️⃣ 绘图
    fig, axes = plt.subplots(len(pollutants), 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Daily Average Pollutant Concentrations (Interpolated)', fontsize=14)

    for i, col in enumerate(pollutants):
        ax = axes[i] if len(pollutants) > 1 else axes
        ax.plot(df_daily[datetime_col], df_daily[col], linewidth=1.0, color='steelblue')
        ax.set_ylabel(col)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 6️⃣ 保存或显示
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved daily average plot (NaN handled) to {save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    cfg = load_global_config()
    data_path = Path(__file__).resolve().parents[2] / cfg["paths"]["processed_data"]
    plot_pollutant_timeseries(
        data_path,
        datetime_col=cfg["data"]["datetime_col"],
        pollutants=cfg["data"]["all_pollutants"],
        save_path=Path(__file__).resolve().parents[2] / cfg["paths"]["plots"] / "pollutant_timeseries.png"
    )
