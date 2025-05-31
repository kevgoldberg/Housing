# Scripts Directory

Place Python scripts for data processing, feature engineering, and modeling here.

- `download_data.py` - download the Kaggle dataset (requires Kaggle API)
- `preprocess_data.py` - run the data cleaning steps from the notebooks.
  Use `--method advanced` for scaling and one-hot encoding.
- `correlation_analysis_runner.py` - perform correlation analysis on the
  processed data. Supports a `--config` option to load settings from
  `config/correlation_analysis_config.yaml`.

All scripts provide `--help` output describing their options.
