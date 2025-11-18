#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_anomalies.py (Multi-Output Version)
=================================================
Performs anomaly detection for EACH target pollutant using a
multi-output regression model.

Pipeline:
 - Residual-based anomalies (Top 1% residuals)
 - Isolation Forest anomaly scores
 - Meteorological interpretation
 - Precision–Recall evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from src.data.data_loader import load_datasets
from src.data.preprocess import load_global_config


# ============================================================
# 1️⃣ Residual-based anomaly detection
# ============================================================
def detect_residual_anomalies(model, X_test, y_test, target):
    """
    Multi-output model: y_pred is shape (N, num_targets)
    """
    y_pred = model.predict(X_test)

    if y_pred.ndim != 2:
        raise ValueError("❌ The model must be MULTI-OUTPUT. y_pred should be 2D.")

    target_idx = list(y_test.columns).index(target)
    y_pred_target = y_pred[:, target_idx]        # prediction for this pollutant
    y_true = y_test[target].values

    residuals = y_true - y_pred_target

    # Top 1% absolute residuals are anomalies
    threshold = np.percentile(np.abs(residuals), 99)
    anomaly_idx = np.where(np.abs(residuals) >= threshold)[0]

    print(f"🔍 [{target}] residual anomalies: {len(anomaly_idx)}")
    return residuals, anomaly_idx


# ============================================================
# 2️⃣ Isolation Forest anomaly score
# ============================================================
def detect_unsupervised_scores(X_test):
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_test)
    return -iso.decision_function(X_test)   # larger score = more abnormal


# ============================================================
# 3️⃣ Interpretation plots
# ============================================================
def interpret_anomalies(df_test_raw, anomaly_idx, datetime_col, target, save_dir):

    df = df_test_raw.copy()
    df["weekday"] = df[datetime_col].dt.day_name()

    anomaly_df = df.iloc[anomaly_idx][
        [datetime_col, "T", "RH", "AH", target, "weekday"]
    ]
    anomaly_df.to_csv(save_dir / f"{target}_anomalies_interpretation.csv", index=False)

    print(f"\n📅 [{target}] Weekday distribution:")
    print(anomaly_df["weekday"].value_counts())

    # Temperature vs pollutant scatter plot
    plt.figure(figsize=(7, 5))
    plt.scatter(df["T"], df[target], alpha=0.2, label="normal")
    plt.scatter(anomaly_df["T"], anomaly_df[target], color="red", label="anomaly")
    plt.xlabel("Temperature (°C)")
    plt.ylabel(target)
    plt.title(f"{target} vs Temperature (Anomalies Highlighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{target}_temp_scatter.png", dpi=300)
    plt.close()


# ============================================================
# 4️⃣ Precision–Recall curve
# ============================================================
def evaluate_precision_recall(residual_scores, unsup_scores, target, save_dir):

    N = len(residual_scores)
    k = max(1, int(N * 0.01))

    gt = np.zeros(N)
    top_idx = np.argsort(residual_scores)[-k:]
    gt[top_idx] = 1

    P, R, _ = precision_recall_curve(gt, unsup_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(R, P, label=target)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve ({target})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{target}_pr_curve.png", dpi=300)
    plt.close()

    # threshold = median score
    pred_labels = unsup_scores > np.median(unsup_scores)
    precision = precision_score(gt, pred_labels)
    recall = recall_score(gt, pred_labels)
    f1 = f1_score(gt, pred_labels)

    print(f"\n📊 [{target}] PR Summary")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")


# ============================================================
# 5️⃣ Run detection for one pollutant
# ============================================================
def run_detection_for_target(model, cfg, X_test, y_test, target):

    print(f"\n======================================")
    print(f"🔎 Running anomaly detection for: {target}")
    print(f"======================================\n")

    save_dir = (
        Path(__file__).resolve().parents[2]
        / cfg["paths"]["plots"]
        / "anomalies"
        / target
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # A. residual-based anomalies
    residuals, residual_idx = detect_residual_anomalies(model, X_test, y_test, target)

    # B. isolation forest score
    unsup_scores = detect_unsupervised_scores(X_test)

    # C. residual plot
    plt.figure(figsize=(10, 4))
    plt.plot(residuals, alpha=0.5)
    plt.scatter(residual_idx, residuals[residual_idx], color="red", label="anomaly")
    plt.title(f"Residuals with Anomalies ({target})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{target}_residuals.png", dpi=300)
    plt.close()

    # D. raw data for interpretation
    root = Path(__file__).resolve().parents[2]
    raw_path = root / cfg["paths"]["processed_data"]

    df_full = pd.read_csv(raw_path, parse_dates=[cfg["data"]["datetime_col"]])
    df_full = df_full.sort_values(cfg["data"]["datetime_col"]).reset_index(drop=True)

    N = len(X_test)
    df_test_raw = df_full.iloc[-N:].reset_index(drop=True)

    cols = [cfg["data"]["datetime_col"], "T", "RH", "AH", target]
    df_test_raw = df_test_raw[cols]

    # E. interpretation
    interpret_anomalies(df_test_raw, residual_idx, cfg["data"]["datetime_col"], target, save_dir)

    # F. PR curve
    evaluate_precision_recall(np.abs(residuals), unsup_scores, target, save_dir)


# ============================================================
# 6️⃣ Multi-target main entry
# ============================================================
def detect_anomalies_multi(model_path, cfg):

    print("🚀 Loading MULTI-OUTPUT regression model...")
    model = load(model_path)

    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(cfg)

    targets = list(y_test.columns)
    print(f"🎯 Targets detected: {targets}")

    for target in targets:
        run_detection_for_target(model, cfg, X_test, y_test, target)

    print("\n🎉 Multi-target anomaly detection complete!")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    cfg = load_global_config()

    # Multi-output model
    model_path = (
        Path(__file__).resolve().parents[2]
        / "results/regression/hourly/best_models"
        / "RandomForestRegressor_best.joblib"
    )

    detect_anomalies_multi(model_path, cfg)
