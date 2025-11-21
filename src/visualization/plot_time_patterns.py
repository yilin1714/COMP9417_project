
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.preprocess import load_global_config

def plot_time_patterns(cfg):

    root_dir = Path(__file__).resolve().parents[2]
    data_path = root_dir / cfg["paths"]["processed_data"]
    save_dir = root_dir / cfg["paths"]["plots"]
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    datetime_col = cfg["data"]["datetime_col"]
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    pollutants = cfg["data"].get("all_pollutants") or [
        col for col in df.columns if col not in [datetime_col]
    ]

    df = df.sort_values(by=datetime_col)

    df[pollutants] = (
        df[pollutants]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    df["Hour"] = df[datetime_col].dt.hour
    df["Month"] = df[datetime_col].dt.month
    df["Weekday"] = df[datetime_col].dt.day_name()

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

    print(f"Saved all time pattern plots to: {save_dir.resolve()}")

if __name__ == "__main__":
    cfg = load_global_config()
    plot_time_patterns(cfg)
