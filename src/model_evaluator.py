"""Utility functions for evaluating and comparing models."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


RMSE_SCORER = make_scorer(_rmse, greater_is_better=False)
MAE_SCORER = make_scorer(mean_absolute_error, greater_is_better=False)
R2_SCORER = make_scorer(r2_score)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def evaluate_models(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """Evaluate multiple models via cross-validation.

    Parameters
    ----------
    models: Dict[str, estimator]
        Mapping of model name to scikit-learn estimator.
    X : DataFrame
        Feature matrix.
    y : Series
        Target vector.
    cv : int, optional
        Number of CV folds, by default 5.

    Returns
    -------
    DataFrame
        DataFrame with mean RMSE, MAE and R^2 for each model.
    """

    records = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={"rmse": RMSE_SCORER, "mae": MAE_SCORER, "r2": R2_SCORER},
            n_jobs=-1,
        )
        record = {
            "model": name,
            "rmse": -scores["test_rmse"].mean(),
            "mae": -scores["test_mae"].mean(),
            "r2": scores["test_r2"].mean(),
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df.sort_values("rmse")
