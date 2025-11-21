
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.preprocess import load_global_config


def analyze_meteorological_relations(cfg):
    root_dir = Path(__file__).resolve().parents[2]
    data_path = root_dir / cfg["paths"]["processed_data"]
    save_dir = root_dir / cfg["paths"]["plots"]
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    datetime_col = cfg["data"]["datetime_col"]
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    pollutants = cfg["data"].get("all_pollutants", [])
    meteo_vars = cfg["data"].get("meteo_vars", ["T", "RH", "AH"])

    pollutants = [p for p in pollutants if p in df.columns]
    meteo_vars = [m for m in meteo_vars if m in df.columns]

    if len(pollutants) == 0:
        print("No pollutant columns found. Check config['data']['all_pollutants']")
        return

    if len(meteo_vars) == 0:
        print("No meteorological variables found. Check config['data']['meteo_vars']")
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_vars = [v for v in pollutants + meteo_vars if v in numeric_cols]

    df[all_vars] = (
        df[all_vars]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    corr_matrix = df[all_vars].corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        square=True
    )
    plt.title("Correlation between Meteorological and Pollutant Variables")
    plt.tight_layout()
    plt.savefig(save_dir / "meteo_pollutant_correlation_heatmap.png", dpi=300)
    plt.close()

    print("Saved: meteo_pollutant_correlation_heatmap.png")

    for p in pollutants:
        for m in meteo_vars:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=m, y=p, alpha=0.4)
            sns.regplot(data=df, x=m, y=p, scatter=False, color="red", ci=None)
            plt.title(f"{p} vs {m}")
            plt.xlabel(m)
            plt.ylabel(p)
            plt.tight_layout()
            plt.savefig(save_dir / f"scatter_{p}_vs_{m}.png", dpi=300)
            plt.close()
            print(f"Saved: scatter_{p}_vs_{m}.png")

    print(f"Saved meteorological relation plots to {save_dir.resolve()}")

if __name__ == "__main__":
    cfg = load_global_config()
    analyze_meteorological_relations(cfg)
