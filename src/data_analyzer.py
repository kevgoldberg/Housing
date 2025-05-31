import pandas as pd
from pathlib import Path
from typing import Union


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of basic statistics for each column in *df*.

    The summary includes the column dtype, number of missing values,
    percentage missing, and number of unique values. For numeric
    columns additional statistics such as mean, standard deviation,
    minimum and maximum are provided.
    """
    summary = []
    for col in df.columns:
        series = df[col]
        info = {
            "column": col,
            "dtype": series.dtype.name,
            "missing": series.isna().sum(),
            "missing_pct": round(series.isna().mean() * 100, 2),
            "unique": series.nunique(dropna=True),
        }
        if pd.api.types.is_numeric_dtype(series):
            info.update(
                mean=round(series.mean(), 3),
                std=round(series.std(), 3),
                min=round(series.min(), 3),
                max=round(series.max(), 3),
            )
        summary.append(info)
    return pd.DataFrame(summary)


def summarize_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file and return :func:`analyze_dataframe` output."""
    path = Path(path)
    df = pd.read_csv(path)
    return analyze_dataframe(df)
