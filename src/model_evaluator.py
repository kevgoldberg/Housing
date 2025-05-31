"""Utility functions for benchmarking regression models."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def benchmark_regressors(
    df: pd.DataFrame,
    target: str = "SalePrice",
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train several regression models and return evaluation metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing features and the target column.
    target : str, default "SalePrice"
        Name of the target variable.
    test_size : float, default 0.2
        Fraction of data used for testing.
    random_state : int, default 42
        Random seed for the train/test split and models.

    Returns
    -------
    pandas.DataFrame
        Table with metrics (RMSE, MAE, R2) sorted by RMSE.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models: Dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }

    results: List[Dict[str, float]] = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return results_df
