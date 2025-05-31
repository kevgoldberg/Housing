# Data Science Project: Kaggle Housing

This repository contains utilities and scripts used to explore and model the Kaggle housing dataset.

## Structure
- `data/raw/` - Place raw Kaggle data here
- `data/processed/` - For cleaned/engineered data
- `notebooks/` - Jupyter notebooks for EDA and modeling
- `scripts/` - Python scripts for data processing and modeling
- `tests/` - Unit tests

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Launch JupyterLab or VS Code to work with the notebooks.

## Data Preprocessing
Run the preprocessing script to convert the raw Kaggle data into a cleaned dataset. This replicates the steps originally performed in the Jupyter notebooks:

```bash
python scripts/preprocess_data.py \
  --input data/raw/train.csv \
  --output data/processed/V1/train_cleaned.csv
```

For a more feature-rich preprocessing pipeline that performs scaling and one-hot encoding, add `--method advanced`:

```bash
python scripts/preprocess_data.py \
  --method advanced \
  --input data/raw/train.csv \
  --output data/processed/V2/train_preprocessed.csv
```

## Correlation Analysis
Use the correlation analysis runner to generate a detailed report of relationships in the processed data:

```bash
python scripts/correlation_analysis_runner.py --config config/correlation_analysis_config.yaml
```

## Running Tests
Run unit tests before committing changes:

```bash
pytest -q
```

See `AGENTS.md` for contribution guidelines.
