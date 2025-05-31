"""Advanced Gradient Boosting Model for Housing Prices

This script implements an end-to-end machine learning pipeline for predicting
house prices. It demonstrates advanced usage of scikit-learn, including

* automatic preprocessing of numeric and categorical features using a
  ``ColumnTransformer``
* a gradient boosting regressor with hyperparameter tuning via
  ``GridSearchCV``
* model evaluation with multiple metrics
* persistent storage of the trained estimator using ``joblib``

The goal is to provide a more sophisticated baseline compared to the simple
linear regression example. While still approachable, the pipeline lays out a
structured framework that can be extended with additional feature engineering or
model types.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the cleaned dataset. Update if your processed data lives elsewhere.
DATA_PATH = Path("..") / ".." / "data" / "processed" / "V2" / "train_advanced_cleaned.csv"

# Location where the best model will be saved.
MODEL_SAVE_PATH = Path("..") / "advanced" / "gradient_boosting_model.pkl"

# Parameters for GridSearchCV. These can be tuned further as needed.
PARAM_GRID = {
    "regressor__n_estimators": [100, 300],
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__max_depth": [3, 5],
}

# Fraction of data to reserve for testing.
TEST_SIZE = 0.2
# Random seed for reproducibility.
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Split a DataFrame into features and target and detect column types."""
    if "SalePrice" not in df.columns:
        raise ValueError("SalePrice column not found in data.")

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return X, y, numeric_features, categorical_features


def _build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Construct the preprocessing and modeling pipeline."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    regressor = GradientBoostingRegressor(random_state=RANDOM_STATE)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", regressor),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------

def main() -> None:
    """Train a gradient boosting model and report detailed metrics."""
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape: {df.shape}")

    X, y, num_features, cat_features = _split_features(df)
    print(f"Numeric features: {len(num_features)} | Categorical features: {len(cat_features)}")

    pipeline = _build_pipeline(num_features, cat_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=5,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
        verbose=2,
    )

    print("Starting hyperparameter search ...")
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")

    best_model = grid.best_estimator_

    print("Evaluating on the held-out test set ...")
    y_pred = best_model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2 : {r2:.3f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
