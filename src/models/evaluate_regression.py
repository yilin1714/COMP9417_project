
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import matplotlib.pyplot as plt
import inspect
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocess import load_global_config

def compute_multi_output_metrics(y_true_df, y_pred):

    metrics = {}

    sig = inspect.signature(mean_squared_error)
    has_squared = ("squared" in sig.parameters)

    for i, col in enumerate(y_true_df.columns):
        y_t = y_true_df[col].values
        y_p = y_pred[:, i]

        mae = mean_absolute_error(y_t, y_p)

        if has_squared:
            rmse = mean_squared_error(y_t, y_p, squared=False)
        else:
            rmse = mean_squared_error(y_t, y_p) ** 0.5

        r2   = r2_score(y_t, y_p)

        metrics[col] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    return metrics

def plot_per_target(y_true, y_pred, target, model_name, save_dir):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor="k")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.title(f"{model_name} — {target}: Predicted vs True")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_{target}_scatter.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(f"{model_name} — {target}: Time-series")
    plt.xlabel("Time Index")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_{target}_timeseries.png")
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=25, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"{model_name} — {target}: Residual Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_{target}_residual_hist.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"{model_name} — {target}: Residual Over Time")
    plt.xlabel("Time Index")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_{target}_residual_time.png")
    plt.close()

def evaluate_for_granularity(cfg, granularity):

    root = Path(__file__).resolve().parents[2]
    split_dir = root / "data/splits"

    # Load multi-output test set
    X_test = pd.read_csv(split_dir / f"X_test_{granularity}.csv")
    y_test_df = pd.read_csv(split_dir / f"y_test_{granularity}.csv")

    print(f"X_test={X_test.shape}, y_test={y_test_df.shape}")

    models_dir = root / "results/regression" / granularity / "best_models"
    plots_dir = root / "results/regression" / granularity / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model_path in models_dir.glob("*_best.joblib"):
        model_name = model_path.stem.replace("_best", "")
        print(f"\nEvaluating: {model_name}")

        model = load(model_path)
        y_pred = model.predict(X_test)

        metrics = compute_multi_output_metrics(y_test_df, y_pred)

        for target, m in metrics.items():
            print(f"   [{target}] MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

            idx = list(y_test_df.columns).index(target)
            plot_per_target(
                y_test_df[target].values,
                y_pred[:, idx],
                target,
                model_name,
                plots_dir
            )

        flat = {"Model": model_name}
        for t, m in metrics.items():
            flat[f"{t}_MAE"] = m["MAE"]
            flat[f"{t}_RMSE"] = m["RMSE"]
            flat[f"{t}_R2"]   = m["R2"]
        results.append(flat)

    out_path = root / "results/regression" / granularity / "test_metrics.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    return results

if __name__ == "__main__":
    cfg = load_global_config()

    evaluate_for_granularity(cfg, "hourly")
    evaluate_for_granularity(cfg, "daily")

    print("\nALL evaluations finished!")
