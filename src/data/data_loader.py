
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def load_datasets(cfg, as_tensor=False):

    data_dir = Path(__file__).resolve().parents[2] / "data/splits"
    print(f"Loading datasets from: {data_dir}")

    X_train = pd.read_csv(data_dir / "X_train_hourly.csv")
    y_train = pd.read_csv(data_dir / "y_train_hourly.csv")
    X_val   = pd.read_csv(data_dir / "X_val_hourly.csv")
    y_val   = pd.read_csv(data_dir / "y_val_hourly.csv")
    X_test  = pd.read_csv(data_dir / "X_test_hourly.csv")
    y_test  = pd.read_csv(data_dir / "y_test_hourly.csv")

    print(f"Loaded datasets: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    if cfg["data"].get("normalize", False):
        print("Applying StandardScaler normalization...")
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val   = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    if as_tensor:
        import torch
        print("Converting to PyTorch tensors...")
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        X_val   = torch.tensor(X_val.values, dtype=torch.float32)
        y_val   = torch.tensor(y_val.values, dtype=torch.float32)
        X_test  = torch.tensor(X_test.values, dtype=torch.float32)
        y_test  = torch.tensor(y_test.values, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    from src.data.preprocess import load_global_config
    cfg = load_global_config()