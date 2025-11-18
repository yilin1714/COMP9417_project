#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classification_models.py
========================
Define and return commonly used classification models:
- Logistic Regression
- Random Forest Classifier
- MLP Classifier

Each model factory returns an sklearn estimator ready to fit().
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ============================================================
# 🎯 Logistic Regression
# ============================================================
def get_logistic_regression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42):
    """
    Return a LogisticRegression model.
    Args:
        C (float): Inverse of regularization strength; smaller values specify stronger regularization.
        max_iter (int): Maximum iterations for convergence.
        solver (str): Algorithm to use ('lbfgs', 'liblinear', etc.)
    """
    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )


# ============================================================
# 🌲 Random Forest Classifier
# ============================================================
def get_random_forest_classifier(
    n_estimators=100, max_depth=None, min_samples_split=2, random_state=42
):
    """
    Return a RandomForestClassifier.
    Args:
        n_estimators (int): Number of trees.
        max_depth (int): Max depth of each tree.
        min_samples_split (int): Minimum samples to split a node.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )


# ============================================================
# 🧠 MLP Classifier
# ============================================================
def get_mlp_classifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=200,
    random_state=42,
):
    """
    Return a Multi-Layer Perceptron classifier.
    Args:
        hidden_layer_sizes (tuple): Number of neurons per layer.
        activation (str): Activation function ('relu', 'tanh', 'logistic').
        solver (str): Optimizer ('adam', 'sgd', 'lbfgs').
        alpha (float): L2 penalty (regularization term).
        learning_rate_init (float): Initial learning rate.
        max_iter (int): Maximum iterations.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state
    )


# ============================================================
# 📦 Model Dispatcher
# ============================================================
def get_classification_models():
    """
    Return a dictionary of model constructors for unified access.
    Example:
        models = get_classification_models()
        clf = models["random_forest"]()
    """
    return {
        "logistic_regression": get_logistic_regression,
        "random_forest": get_random_forest_classifier,
        "mlp": get_mlp_classifier,
    }

if __name__ == "__main__":
    print(get_classification_models())