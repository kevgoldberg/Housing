#!/usr/bin/env python3
"""Compare multiple models using cross-validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root / "src"))

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from model_evaluator import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model performance")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/V2/train_advanced_cleaned.csv",
        help="Path to the preprocessed training data",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="SalePrice",
        help="Name of the target variable",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target} not found")

    X = df.drop(args.target, axis=1)
    y = df[args.target]

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = evaluate_models(models, X, y, cv=args.cv)
    print("\nModel comparison (lower RMSE/MAE is better, higher R2 is better):")
    print(results.to_string(index=False, formatters={"rmse": "{:.2f}".format, "mae": "{:.2f}".format, "r2": "{:.3f}".format}))


if __name__ == "__main__":
    main()
