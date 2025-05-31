import pandas as pd
from src.data_analyzer import analyze_dataframe


def test_analyze_dataframe_basic():
    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": ["x", "y", "y"],
    })
    summary = analyze_dataframe(df)
    # ensure one missing value in column A
    a_row = summary[summary["column"] == "A"].iloc[0]
    assert a_row["missing"] == 1
    # categorical column unique count
    b_row = summary[summary["column"] == "B"].iloc[0]
    assert b_row["unique"] == 2
