#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_regression.py (Multi-Output Version)
===========================================
Trains 3 regression models (Linear, RF, MLP)
on:
    - hourly_features.csv
    - daily_features.csv

Each model predicts ALL target pollutants simultaneously
using MultiOutputRegressor.

Saves:
    results/regression/<granularity>/best_models/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump

from src.data.preprocess import load_global_config
from src.data.split_dataset import split_dataset

from src.models.regression_models import (
    get_linear_regression,
    get_random_forest_regressor,
    get_mlp_regressor,
)

# ============================================================
# 🧮 Evaluation helper
# Multi-output: metrics averaged over all targets
# ============================================================
def evaluate_regression(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ============================================================
# 🚀 Train ONE granularity (hourly/daily)
# ============================================================
def train_for_feature_file(cfg, features_file):
    print(f"\n🚀 Training MULTI-OUTPUT regression models using: {features_file}")

    root = Path(__file__).resolve().parents[2]

    prefix = features_file.replace("_features.csv", "").replace(".csv", "")
    granularity = prefix  # hourly / daily
    print(f"📌 Detected granularity: {granularity}")

    # 1) Split dataset
    split_dataset(features_file, cfg)

    # 2) Load split data
    split_dir = root / "data" / "splits"

    X_train = pd.read_csv(split_dir / f"X_train_{prefix}.csv")
    y_train = pd.read_csv(split_dir / f"y_train_{prefix}.csv")      # MULTI-OUTPUT
    X_val = pd.read_csv(split_dir / f"X_val_{prefix}.csv")
    y_val = pd.read_csv(split_dir / f"y_val_{prefix}.csv")          # MULTI-OUTPUT

    print(f"📊 Loaded: Train={len(X_train)}, Val={len(X_val)}, Targets={list(y_train.columns)}")

    # 3) Define base regressors
    model_builders = {
        "LinearRegression": get_linear_regression,
        "RandomForestRegressor": get_random_forest_regressor,
        "MLPRegressor": get_mlp_regressor,
    }

    param_grids = {
        "LinearRegression": [{}],
        "RandomForestRegressor": [
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5]
            }
        ],
        "MLPRegressor": [
            {
                "hidden_layer_sizes": [(64, 32), (128, 64)],
                "max_iter": [300, 500],
                "learning_rate_init": [0.001, 0.0005]
            }
        ],
    }

    best_models = {}
    results = []

    # 4) Hyperparameter tuning (MultiOutputRegressor)
    for name, builder in model_builders.items():
        print(f"\n⚙️ Training {name} (Multi-output)...")

        # expand grid
        grids = []
        for grid in param_grids.get(name, [{}]):
            if not grid:
                grids.append({})
            else:
                keys, values = zip(*grid.items())
                for combo in product(*values):
                    grids.append(dict(zip(keys, combo)))

        best_r2 = -np.inf
        best_params = None
        best_model = None
        best_metrics = None

        for params in grids:
            base_model = builder(**params)
            model = MultiOutputRegressor(base_model)

            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            val_metrics = evaluate_regression(y_val, val_pred)

            print(f"   params={params} | R²={val_metrics['R2']:.3f}")

            if val_metrics["R2"] > best_r2:
                best_r2 = val_metrics["R2"]
                best_params = params
                best_model = model
                best_metrics = val_metrics

        best_models[name] = best_model
        results.append({
            "Model": name,
            "Best_Params": best_params,
            **{f"Val_{k}": v for k, v in best_metrics.items()},
        })

    # 5) Save best models
    results_root = root / "results" / "regression"
    models_dir = results_root / granularity / "best_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in best_models.items():
        dump(model, models_dir / f"{name}_best.joblib")

    print(f"💾 Saved MULTI-OUTPUT models to: {models_dir}")

    # 6) Save validation metrics
    result_df = pd.DataFrame(results)
    result_dir = results_root / granularity
    result_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_dir / "regression_val_metrics.csv", index=False)

    print(f"📊 Validation metrics saved to: {result_dir}")

    return result_df


# ============================================================
# 🚀 Train both hourly + daily
# ============================================================
if __name__ == "__main__":
    cfg = load_global_config()

    feature_dir = Path(__file__).resolve().parents[2] / "data" / "features"

    feature_files = [
        "hourly_features.csv",
        "daily_features.csv",
    ]

    for f in feature_files:
        if (feature_dir / f).exists():
            train_for_feature_file(cfg, f)
        else:
            print(f"⚠️ Skip {f}: file not found.")
