
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocess import load_global_config

def discretise_co(value):
    if value < 1.5:
        return "low"
    elif value < 2.5:
        return "mid"
    else:
        return "high"


def add_features(df, cfg):
    datetime_col = cfg["data"]["datetime_col"]
    target = cfg["classification"]["base_target"]
    fe = cfg["data"]["feature_engineering"]

    lookback = fe["lookback"]
    roll_windows = fe["roll_windows"]
    use_time = fe["include_time_features"]
    use_cyclic = fe["use_cyclical_encoding"]

    for lag in range(1, lookback + 1):
        df[f"{target}_t-{lag}h"] = df[target].shift(lag)

    for w in roll_windows:
        df[f"{target}_roll_mean_{w}h"] = df[target].rolling(w).mean()
        df[f"{target}_roll_std_{w}h"] = df[target].rolling(w).std()

    if use_time:
        df["weekday"] = df[datetime_col].dt.dayofweek
        df["month"] = df[datetime_col].dt.month
        df["hour"] = df[datetime_col].dt.hour

        if use_cyclic:
            df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def add_class_labels(df, cfg):
    target = cfg["classification"]["base_target"]
    horizons = cfg["classification"]["horizons"]

    df[f"{target}_class_t+0"] = df[target].apply(discretise_co)

    for h in horizons:
        df[f"{target}_t+{h}h"] = df[target].shift(-h)
        df[f"{target}_class_t+{h}"] = df[f"{target}_t+{h}h"].apply(discretise_co)

    return df


def create_classification_features(input_path, output_path, cfg):
    print("\nCreating HOURLY CLASSIFICATION FEATURES...")

    datetime_col = cfg["data"]["datetime_col"]

    df = pd.read_csv(input_path, parse_dates=[datetime_col]).sort_values(datetime_col)

    drop_cols = cfg["data"].get("drop_columns", [])
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df = add_class_labels(df, cfg)

    df = add_features(df, cfg)

    future_label_cols = [
        f"{cfg['classification']['base_target']}_class_t+{h}"
        for h in cfg["classification"]["horizons"]
    ]
    df = df.dropna(subset=future_label_cols).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved classification features → {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    cfg = load_global_config()
    root = Path(__file__).resolve().parents[2]
    input_path = root / cfg["paths"]["processed_data"]
    output_path = root / "data/features/classification_features.csv"

    create_classification_features(input_path, output_path, cfg)

    print("\nClassification feature engineering complete!")
