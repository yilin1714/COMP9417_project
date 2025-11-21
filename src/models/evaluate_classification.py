
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from src.data.preprocess import load_global_config

def compute_classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def plot_confusion_matrix(y_true_enc, y_pred, model_name, save_dir):
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true_enc, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["low", "mid", "high"],
        yticklabels=["low", "mid", "high"]
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()

def plot_pred_vs_true(y_true_enc, y_pred, model_name, save_dir):
    plt.figure(figsize=(12, 4))
    plt.plot(y_true_enc, label="True", linewidth=2)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title(f"{model_name} - Predicted vs True (timeline)")
    plt.xlabel("Sample index (time)")
    plt.ylabel("Class (0=low, 1=mid, 2=high)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_timeline.png", dpi=200)
    plt.close()

def evaluate_horizon(cfg, horizon):

    root = Path(__file__).resolve().parents[2]
    split_dir = root / "data/splits"
    model_dir = root / f"results/classification/t+{horizon}"


    X_test = pd.read_csv(split_dir / "X_test_classification.csv")

    y_test = pd.read_csv(
        split_dir / f"y_test_classification_t+{horizon}.csv"
    ).iloc[:, 0].values

    mapping = {"low": 0, "mid": 1, "high": 2}
    y_test_enc = np.vectorize(mapping.get)(y_test)

    plot_dir = model_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("\nEvaluating Naïve Baseline...")

    baseline_path = split_dir / "y_test_classification_t+0.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(
            "y_test_classification_t+0.csv not found"
        )

    y_test_t0 = pd.read_csv(baseline_path).iloc[:, 0].map(mapping).values
    y_pred_baseline = y_test_t0.copy()

    baseline_metrics = compute_classification_metrics(y_test_enc, y_pred_baseline)

    print(f"Baseline Acc={baseline_metrics['Accuracy']:.3f}, "
          f"Precision={baseline_metrics['Precision']:.3f}, "
          f"Recall={baseline_metrics['Recall']:.3f}, "
          f"F1={baseline_metrics['F1']:.3f}")

    plot_confusion_matrix(
        y_test_enc, y_pred_baseline,
        model_name=f"baseline_t+{horizon}",
        save_dir=plot_dir
    )
    plot_pred_vs_true(
        y_test_enc, y_pred_baseline,
        model_name=f"baseline_t+{horizon}",
        save_dir=plot_dir
    )

    results.append({
        "Model": "Baseline",
        "Horizon": horizon,
        **baseline_metrics
    })

    for model_path in model_dir.glob("*.joblib"):
        model_name = model_path.stem.replace("_best", "")
        print(f"\nModel: {model_name}")

        model = load(model_path)
        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test_enc, y_pred)
        print(f"Acc={metrics['Accuracy']:.3f}, "
              f"Prec={metrics['Precision']:.3f}, Recall={metrics['Recall']:.3f}, "
              f"F1={metrics['F1']:.3f}")

        plot_confusion_matrix(y_test_enc, y_pred, model_name, plot_dir)
        plot_pred_vs_true(y_test_enc, y_pred, model_name, plot_dir)

        results.append({
            "Model": model_name,
            "Horizon": horizon,
            **metrics
        })

    df = pd.DataFrame(results)
    df.to_csv(model_dir / f"metrics_t+{horizon}.csv", index=False)

    print(f"Saved metrics → {model_dir / f'metrics_t+{horizon}.csv'}")
    return df

if __name__ == "__main__":
    cfg = load_global_config()
    horizons = cfg["classification"]["horizons"]

    all_results = []
    for h in horizons:
        df = evaluate_horizon(cfg, h)
        all_results.append(df)

    final = pd.concat(all_results, ignore_index=True)

    root = Path(__file__).resolve().parents[2]
    summary_path = root / "results/classification/classification_test_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(summary_path, index=False)

    print("\nAll classification evaluations finished!")
    print("Summary saved to:", summary_path)
