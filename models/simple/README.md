# Simple Models

This directory contains example baseline models for predicting house prices.

Available scripts:

- `linear_regression.py` - Ordinary least squares regression.
- `decision_tree_regression.py` - Decision tree regressor.
- `random_forest_regression.py` - Random forest regressor.

For an easy way to see which of these performs best on your data, run the
`scripts/compare_models.py` utility after preprocessing. It reports cross-
validated RMSE, MAE and R² scores for each model.
