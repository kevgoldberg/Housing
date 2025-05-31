import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.model_evaluator import evaluate_models


def test_evaluate_models_basic():
    X, y = make_regression(n_samples=50, n_features=4, noise=0.1, random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    models = {
        "lr": LinearRegression(),
        "tree": DecisionTreeRegressor(random_state=42),
    }

    results = evaluate_models(models, X, y, cv=3)
    assert set(results.columns) == {"model", "rmse", "mae", "r2"}
    assert len(results) == 2

