
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.preprocess import load_global_config

def plot_pollutant_timeseries(data_path, datetime_col='Date', pollutants=None, save_path=None):

    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    df = pd.read_csv(data_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    if pollutants is None:
        pollutants = [col for col in df.columns if col not in [datetime_col]]

    df_daily = (
        df.set_index(datetime_col)
          .resample('D')[pollutants]
          .mean()
          .reset_index()
    )

    df_daily = df_daily.interpolate(method='linear').ffill()

    fig, axes = plt.subplots(len(pollutants), 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Daily Average Pollutant Concentrations (Interpolated)', fontsize=14)

    for i, col in enumerate(pollutants):
        ax = axes[i] if len(pollutants) > 1 else axes
        ax.plot(df_daily[datetime_col], df_daily[col], linewidth=1.0, color='steelblue')
        ax.set_ylabel(col)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved daily average plot (NaN handled) to {save_path}")
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
