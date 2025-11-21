
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_global_config(path="../../config/global.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess(input_path, output_path, cfg):

    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path, sep=";", decimal=",", low_memory=False)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    missing_value = cfg["data"].get("missing_value", -200)
    df.replace(missing_value, np.nan, inplace=True)

    if "Date" in df.columns and "Time" in df.columns:
        df["DateTime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H.%M.%S",
            errors="coerce"
        )
        df.drop(columns=["Date", "Time"], inplace=True)

    df.dropna(subset=["DateTime"], inplace=True)
    df.dropna(
        how="all",
        subset=[col for col in df.columns if col != "DateTime"],
        inplace=True
    )

    df.sort_values(by="DateTime", inplace=True)
    df.drop_duplicates(subset=["DateTime"], keep="first", inplace=True)

    if cfg["data"].get("smoothing", False):
        window = cfg["data"].get("smoothing_window", 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            df[col] = df[col].rolling(window=window, min_periods=1, center=True).mean()

        print(f"Applied {window}-step moving average smoothing.")

    if cfg["data"].get("handle_outliers", False):
        method = cfg["data"].get("outlier_method", "zscore").lower()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "zscore":
            for col in numeric_cols:
                mean, std = df[col].mean(), df[col].std()
                outliers = (df[col] - mean).abs() > 3 * std
                df.loc[outliers, col] = np.nan
            print("Outliers handled using Z-score method (|z| > 3).")

        elif method == "iqr":
            for col in numeric_cols:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
                df.loc[outliers, col] = np.nan
            print("Outliers handled using IQR method.")

    df.reset_index(drop=True, inplace=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to: {output_path}")
    print(f"Total records: {len(df)} | Columns: {len(df.columns)}")

if __name__ == "__main__":
    cfg = load_global_config()
    input_path = Path(__file__).resolve().parents[2] / cfg["paths"]["raw_data"]
    output_path = Path(__file__).resolve().parents[2] / cfg["paths"]["processed_data"]
    preprocess(input_path, output_path, cfg)
