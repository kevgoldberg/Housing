# Advanced Models

This directory contains modeling scripts that go beyond simple linear models.

- **`gradient_boosting_model.py`** – Implements an end-to-end pipeline using
  scikit-learn's `GradientBoostingRegressor` with hyperparameter tuning. The
  script automatically preprocesses numeric and categorical features,
  performs grid search cross‑validation, reports several metrics, and saves the
  resulting model to `gradient_boosting_model.pkl`.

Additional complex models can be added here following the same pattern.
