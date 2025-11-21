
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.preprocess import load_global_config


def analyze_pollutant_correlation(data_path, datetime_col='Date', pollutants=None, save_path=None):

    df = pd.read_csv(data_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    if pollutants is None:
        pollutants = [col for col in df.columns if col not in [datetime_col]]

    df_daily = (
        df.set_index(datetime_col)
          .resample('D')[pollutants]
          .mean()
          .interpolate(method='linear')
          .ffill()
          .bfill()
    )

    corr_matrix = df_daily.corr(method='pearson')

    print("Pearson Correlation Matrix:")
    print(corr_matrix.round(3))

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
        print(f"Saved correlation heatmap to {save_path}")
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
