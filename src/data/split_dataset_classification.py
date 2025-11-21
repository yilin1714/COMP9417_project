
import pandas as pd
from pathlib import Path
from src.data.preprocess import load_global_config


def split_dataset_classification(cfg, val_ratio=0.1):
    root = Path(__file__).resolve().parents[2]
    datetime_col = cfg["data"]["datetime_col"]
    base_target = cfg["classification"]["base_target"]
    horizons = cfg["classification"]["horizons"]

    feature_path = root / "data/features/classification_features.csv"
    print(f"\nLoading classification features: {feature_path}")

    if not feature_path.exists():
        raise FileNotFoundError("classification_features.csv not found")

    df = pd.read_csv(feature_path, parse_dates=[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    df = df.dropna(axis=1, how="all")

    train_val_df = df[df[datetime_col].dt.year == 2004]
    test_df = df[df[datetime_col].dt.year == 2005]

    val_start = int(len(train_val_df) * (1 - val_ratio))
    train_df = train_val_df.iloc[:val_start]
    val_df = train_val_df.iloc[val_start:]

    train_df = train_df.ffill().bfill()
    val_df = val_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    class_cols = [f"{base_target}_class_t+{h}" for h in horizons]

    base_col = f"{base_target}_class_t+0"
    class_cols.append(base_col)

    feature_cols = [
        c for c in df.columns
        if c not in class_cols + [datetime_col]
    ]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    out_dir = root / "data/splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out_dir / "X_train_classification.csv", index=False)
    X_val.to_csv(out_dir / "X_val_classification.csv", index=False)
    X_test.to_csv(out_dir / "X_test_classification.csv", index=False)

    if base_col not in df.columns:
        raise ValueError(
            f"not found {base_col}"
        )

    train_df[base_col].to_csv(out_dir / "y_train_classification_t+0.csv", index=False)
    val_df[base_col].to_csv(out_dir / "y_val_classification_t+0.csv", index=False)
    test_df[base_col].to_csv(out_dir / "y_test_classification_t+0.csv", index=False)

    for h in horizons:
        col_name = f"{base_target}_class_t+{h}"

        train_df[col_name].to_csv(out_dir / f"y_train_classification_t+{h}.csv", index=False)
        val_df[col_name].to_csv(out_dir / f"y_val_classification_t+{h}.csv", index=False)
        test_df[col_name].to_csv(out_dir / f"y_test_classification_t+{h}.csv", index=False)

    print("Classification dataset split complete!")
    return X_train, X_val, X_test

if __name__ == "__main__":
    cfg = load_global_config()
    split_dataset_classification(cfg)
    print("\nAll classification splits completed!")
