
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump

from src.data.preprocess import load_global_config
from src.models.classification_models import (
    get_logistic_regression,
    get_random_forest_classifier,
    get_mlp_classifier,
)

def evaluate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def encode_labels(y):
    mapping = {"low": 0, "mid": 1, "high": 2}
    return np.vectorize(mapping.get)(np.array(y))

def train_for_horizon(cfg, horizon):

    root = Path(__file__).resolve().parents[2]
    split_dir = root / "data/splits"

    X_train = pd.read_csv(split_dir / "X_train_classification.csv")
    X_val   = pd.read_csv(split_dir / "X_val_classification.csv")

    y_train = pd.read_csv(split_dir / f"y_train_classification_t+{horizon}.csv").iloc[:, 0]
    y_val   = pd.read_csv(split_dir / f"y_val_classification_t+{horizon}.csv").iloc[:, 0]

    y_train = encode_labels(y_train)
    y_val   = encode_labels(y_val)

    model_builders = {
        "LogisticRegression": get_logistic_regression,
        "RandomForest": get_random_forest_classifier,
        "MLP": get_mlp_classifier,
    }

    param_grids = {
        "LogisticRegression": [{"C": [0.1, 1.0, 10.0], "max_iter": [500]}],
        "RandomForest": [
            {"n_estimators": [100, 200], "max_depth": [None, 10], "min_samples_split": [2, 5]}
        ],
        "MLP": [
            {"hidden_layer_sizes": [(64, 32), (128, 64)],
             "max_iter": [400],
             "learning_rate_init": [0.001]}
        ]
    }

    best_models = {}
    results = []

    for name, builder in model_builders.items():
        print(f"\nTraining {name} for t+{horizon}...")

        best_acc = -np.inf
        best_params = None
        best_model = None
        best_val = None

        grid_list = param_grids[name]
        combos = [
            dict(zip(g.keys(), combo))
            for g in grid_list
            for combo in product(*g.values())
        ]

        for params in combos:
            model = builder(**params)
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            metrics = evaluate_metrics(y_val, val_pred)

            print(f"   Params={params}, Acc={metrics['Accuracy']:.3f}")

            if metrics["Accuracy"] > best_acc:
                best_acc = metrics["Accuracy"]
                best_params = params
                best_model = model
                best_val = metrics

        print(f"Best {name} params: {best_params}")

        best_models[name] = best_model
        results.append({
            "Horizon": f"t+{horizon}",
            "Model": name,
            **best_val
        })

    model_dir = root / f"results/classification/t+{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in best_models.items():
        dump(model, model_dir / f"{name}_best.joblib")

    print(f"Saved best models → {model_dir}")

    return pd.DataFrame(results)

def train_all_classification(cfg):
    horizons = cfg["classification"]["horizons"]

    all_results = []

    for h in horizons:
        df = train_for_horizon(cfg, h)
        all_results.append(df)

    final = pd.concat(all_results, ignore_index=True)

    root = Path(__file__).resolve().parents[2]
    save_path = root / "results/classification/classification_val_summary.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    final.to_csv(save_path, index=False)
    print("\nSaved summary:", save_path)

    return final

if __name__ == "__main__":
    cfg = load_global_config()
    train_all_classification(cfg)
