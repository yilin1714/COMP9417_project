#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regression_models.py
========================
Define regression models for predicting pollutant concentrations
(e.g., CO(GT), NMHC(GT), C6H6(GT), NOx(GT), NO2(GT)).

Each model can be accessed individually or together via get_regression_models().
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# ============================================================
# 📘 1️⃣ Linear Regression
# ============================================================
def get_linear_regression():
    """Return a LinearRegression model (baseline interpretable model)."""
    return LinearRegression()


# ============================================================
# 🌲 2️⃣ Random Forest Regressor
# ============================================================
def get_random_forest_regressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    verbose=False,
):
    """Return a RandomForestRegressor with configurable parameters."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )


# ============================================================
# ⚡ 3️⃣ MLP Regressor
# ============================================================
def get_mlp_regressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,   # ✅ ← 新增参数
    max_iter=500,
    random_state=42,
    verbose=False,
):
    """Return a Multi-Layer Perceptron Regressor (neural network)."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,   # ✅ ← 同步传给模型
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )

# ============================================================
# 🧩 统一接口：返回所有模型
# ============================================================
def get_regression_models():
    """Return a dictionary containing all regression models."""
    models = {
        "LinearRegression": get_linear_regression(),
        "RandomForestRegressor": get_random_forest_regressor(verbose=True),
        "MLPRegressor": get_mlp_regressor(verbose=True),
    }
    print(f"✅ Loaded {len(models)} regression models: {list(models.keys())}")
    return models


# ============================================================
# 🚀 Main test (for debugging)
# ============================================================
if __name__ == "__main__":
    models = get_regression_models()
    for name, model in models.items():
        print(f"{name}: {model}")
