
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def get_linear_regression():

    return LinearRegression()

def get_random_forest_regressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    verbose=False,
):

    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

def get_mlp_regressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    verbose=False,
):

    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )

def get_regression_models():

    models = {
        "LinearRegression": get_linear_regression(),
        "RandomForestRegressor": get_random_forest_regressor(verbose=True),
        "MLPRegressor": get_mlp_regressor(verbose=True),
    }
    print(f"Loaded {len(models)} regression models: {list(models.keys())}")
    return models

if __name__ == "__main__":
    models = get_regression_models()
    for name, model in models.items():
        print(f"{name}: {model}")
