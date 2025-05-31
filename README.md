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
