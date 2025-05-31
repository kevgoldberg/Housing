"""Data cleaning utilities for the Kaggle housing dataset."""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def simple_preprocess(input_path: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
    """Simple preprocessing routine used in the original notebook.

    This function fills missing numeric values with the median of each column,
    fills missing categorical values with the mode, encodes categorical
    variables as integer codes and saves the cleaned DataFrame.

    Parameters
    ----------
    input_path : str or Path
        Path to the raw CSV file.
    output_path : str or Path
        Where the cleaned CSV will be written.

    Returns
    -------
    pandas.DataFrame
        The cleaned DataFrame.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    # Fill numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical columns and encode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype("category").cat.codes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
