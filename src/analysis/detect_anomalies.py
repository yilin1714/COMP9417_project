
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from src.data.data_loader import load_datasets
from src.data.preprocess import load_global_config


def detect_residual_anomalies(model, X_test, y_test, target):
    y_pred = model.predict(X_test)

    if y_pred.ndim != 2:
        raise ValueError("The model must be MULTI-OUTPUT. y_pred should be 2D.")

    target_idx = list(y_test.columns).index(target)
    y_pred_target = y_pred[:, target_idx]
    y_true = y_test[target].values

    residuals = y_true - y_pred_target

    threshold = np.percentile(np.abs(residuals), 99)
    anomaly_idx = np.where(np.abs(residuals) >= threshold)[0]

    print(f"[{target}] residual anomalies: {len(anomaly_idx)}")
    return residuals, anomaly_idx


def detect_unsupervised_scores(X_test):
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_test)
    return -iso.decision_function(X_test)


def interpret_anomalies(df_test_raw, anomaly_idx, datetime_col, target, save_dir):

    df = df_test_raw.copy()
    df["weekday"] = df[datetime_col].dt.day_name()

    anomaly_df = df.iloc[anomaly_idx][
        [datetime_col, "T", "RH", "AH", target, "weekday"]
    ]
    anomaly_df.to_csv(save_dir / f"{target}_anomalies_interpretation.csv", index=False)

    print(f"\n[{target}] Weekday distribution:")
    print(anomaly_df["weekday"].value_counts())

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

    pred_labels = unsup_scores > np.median(unsup_scores)
    precision = precision_score(gt, pred_labels)
    recall = recall_score(gt, pred_labels)
    f1 = f1_score(gt, pred_labels)

    print(f"\n[{target}] PR Summary")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")


def run_detection_for_target(model, cfg, X_test, y_test, target):

    save_dir = (
        Path(__file__).resolve().parents[2]
        / cfg["paths"]["plots"]
        / "anomalies"
        / target
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    residuals, residual_idx = detect_residual_anomalies(model, X_test, y_test, target)

    unsup_scores = detect_unsupervised_scores(X_test)

    plt.figure(figsize=(10, 4))
    plt.plot(residuals, alpha=0.5)
    plt.scatter(residual_idx, residuals[residual_idx], color="red", label="anomaly")
    plt.title(f"Residuals with Anomalies ({target})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{target}_residuals.png", dpi=300)
    plt.close()

    root = Path(__file__).resolve().parents[2]
    raw_path = root / cfg["paths"]["processed_data"]

    df_full = pd.read_csv(raw_path, parse_dates=[cfg["data"]["datetime_col"]])
    df_full = df_full.sort_values(cfg["data"]["datetime_col"]).reset_index(drop=True)

    N = len(X_test)
    df_test_raw = df_full.iloc[-N:].reset_index(drop=True)

    cols = [cfg["data"]["datetime_col"], "T", "RH", "AH", target]
    df_test_raw = df_test_raw[cols]

    interpret_anomalies(df_test_raw, residual_idx, cfg["data"]["datetime_col"], target, save_dir)

    evaluate_precision_recall(np.abs(residuals), unsup_scores, target, save_dir)


def detect_anomalies_multi(model_path, cfg):

    print("Loading MULTI-OUTPUT regression model...")
    model = load(model_path)

    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(cfg)

    targets = list(y_test.columns)
    print(f"Targets detected: {targets}")

    for target in targets:
        run_detection_for_target(model, cfg, X_test, y_test, target)

    print("\nMulti-target anomaly detection complete!")


if __name__ == "__main__":
    cfg = load_global_config()

    model_path = (
        Path(__file__).resolve().parents[2]
        / "results/regression/hourly/best_models"
        / "RandomForestRegressor_best.joblib"
    )

    detect_anomalies_multi(model_path, cfg)
