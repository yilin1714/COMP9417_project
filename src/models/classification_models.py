
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_logistic_regression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42):

    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )

def get_random_forest_classifier(
    n_estimators=100, max_depth=None, min_samples_split=2, random_state=42
):

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )

def get_mlp_classifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=200,
    random_state=42,
):

    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state
    )

def get_classification_models():

    return {
        "logistic_regression": get_logistic_regression,
        "random_forest": get_random_forest_classifier,
        "mlp": get_mlp_classifier,
    }

if __name__ == "__main__":
    print(get_classification_models())