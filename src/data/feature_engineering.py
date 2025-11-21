
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocess import load_global_config


def create_features(input_path, output_path, cfg):

    print(f"Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=[cfg["data"]["datetime_col"]])
    df.sort_values(cfg["data"]["datetime_col"], inplace=True)

    datetime_col = cfg["data"]["datetime_col"]
    targets = cfg["data"]["targets"]
    fe_params = cfg["data"]["feature_engineering"]
    horizons = cfg.get("prediction", {}).get("horizons", [1, 6, 12, 24])

    lookback = fe_params.get("lookback", 3)
    roll_windows = fe_params.get("roll_windows", [3, 6, 12])
    include_time_features = fe_params.get("include_time_features", True)
    use_cyclical_encoding = fe_params.get("use_cyclical_encoding", True)

    drop_cols = cfg["data"].get("drop_columns", [])
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    print(f"Generating lag features (lookback={lookback}) for all variables...")
    lag_features = {}
    for lag in range(1, lookback + 1):
        for tgt in targets:
            lag_features[f"{tgt}_t-{lag}"] = df[tgt].shift(lag)

    print(f"Generating rolling mean/std features for targets: {targets}")
    roll_features = {}
    for tgt in targets:
        for w in roll_windows:
            roll_features[f"{tgt}_roll_mean_{w}h"] = df[tgt].rolling(window=w).mean()
            roll_features[f"{tgt}_roll_std_{w}h"] = df[tgt].rolling(window=w).std()

    df = pd.concat([df, pd.DataFrame(lag_features), pd.DataFrame(roll_features)], axis=1)
    df = df.copy()

    if include_time_features:
        print("Adding time-based features...")
        df["hour"] = df[datetime_col].dt.hour
        df["weekday"] = df[datetime_col].dt.dayofweek
        df["month"] = df[datetime_col].dt.month

        if use_cyclical_encoding:
            print("Applying cyclical encoding for time features...")
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    print(f"Generating future targets for horizons: {horizons} hours")
    for tgt in targets:
        for h in horizons:
            df[f"{tgt}_t+{h}"] = df[tgt].shift(-h)

    nan_count = df.isna().sum().sum()
    print(f"Keeping NaN rows ({nan_count:,} NaN values exist after feature creation).")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature file saved to: {output_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    cfg = load_global_config()
    input_path = Path(__file__).resolve().parents[2] / cfg["paths"]["processed_data"]
    output_path = Path(__file__).resolve().parents[2] / cfg["paths"]["features_data"]

    print("\nStarting Feature Engineering Pipeline (multi-horizon)...")
    create_features(input_path, output_path, cfg)
    print("All features & targets generated successfully!")
