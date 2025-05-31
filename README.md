# Data Science Project: Kaggle Housing

## Structure
- `data/raw/` - Place raw Kaggle data here
- `data/processed/` - For cleaned/engineered data
- `notebooks/` - Jupyter notebooks for EDA and modeling
- `scripts/` - Python scripts for data processing and modeling
- `tests/` - Unit tests

## Setup
- Use `requirements.txt` to install dependencies
- Use Jupyter or VS Code for notebooks

## Data Preprocessing

Run the preprocessing script to convert the raw Kaggle data into a cleaned
dataset. This replicates the steps originally performed in the Jupyter
notebooks:

```bash
python scripts/preprocess_data.py \
  --input data/raw/train.csv \
  --output data/processed/V1/train_cleaned.csv
```

For a more feature-rich preprocessing pipeline that performs scaling and
one-hot encoding, add `--method advanced`:

```bash
python scripts/preprocess_data.py \
  --method advanced \
  --input data/raw/train.csv \
  --output data/processed/V2/train_preprocessed.csv
```

Use the `--analyze` flag to display a quick summary of the dataset before
running the preprocessing routine:

```bash
python scripts/preprocess_data.py \
  --analyze \
  --input data/raw/train.csv \
  --output data/processed/V1/train_cleaned.csv
```

## Correlation Analysis

Use the correlation analysis runner to explore relationships between variables.
The script can optionally save a JSON report of the results:

```bash
python scripts/correlation_analysis_runner.py \
  --data data/processed/V1/train_cleaned.csv \
  --target SalePrice \
  --output-report analysis_report.json
```

Add `--quick` to perform only a correlation matrix and high-correlation search.

## Running Tests

Run the unit tests with `pytest`:

```bash
pytest -q
```
