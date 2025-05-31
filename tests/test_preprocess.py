import pandas as pd
from src.data_cleaner import simple_preprocess, advanced_preprocess
from pathlib import Path


def test_simple_preprocess(tmp_path):
    data = pd.DataFrame({
        "A": [1, 2, None],
        "B": ["x", None, "y"],
    })
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    data.to_csv(input_csv, index=False)

    cleaned = simple_preprocess(input_csv, output_csv)

    # numeric column filled with median -> 1,2 => median 1.5 => [1,2,1.5]
    assert cleaned["A"].isna().sum() == 0
    assert output_csv.exists()


def test_advanced_preprocess(tmp_path):
    data = pd.DataFrame({
        "A": [1.0, None, 3.0],
        "B": ["x", None, "y"],
    })
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    data.to_csv(input_csv, index=False)

    cleaned = advanced_preprocess(input_csv, output_csv)

    # All values should be numeric with no missing entries
    assert cleaned.isna().sum().sum() == 0
    # one-hot encoded columns should be created for B
    b_cols = [c for c in cleaned.columns if c.startswith("B_")]
    assert len(b_cols) == 2
    assert output_csv.exists()

