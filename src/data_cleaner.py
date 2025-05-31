"""Data cleaning utilities for the Kaggle housing dataset."""
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def advanced_preprocess(input_path: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
    """Advanced preprocessing using scikit-learn pipelines.

    This routine performs the following steps:

    - Numeric features: impute missing values with the median and standardize
      with :class:`~sklearn.preprocessing.StandardScaler`.
    - Categorical features: impute missing values with the most frequent value
      and one-hot encode using :class:`~sklearn.preprocessing.OneHotEncoder`.

    The resulting fully numeric DataFrame is saved to ``output_path``.

    Parameters
    ----------
    input_path : str or Path
        Path to the raw CSV file.
    output_path : str or Path
        Where the processed CSV will be written.

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame with scaled numeric columns and one-hot encoded
        categorical columns.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    num_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: List[str] = df.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if num_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, num_cols))

    if cat_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
        transformers.append(("cat", categorical_pipe, cat_cols))

    if not transformers:
        processed_df = df.copy()
    else:
        preprocessor = ColumnTransformer(transformers)
        processed_array = preprocessor.fit_transform(df)

        feature_names: List[str] = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            oh_encoder = preprocessor.named_transformers_["cat"]["onehot"]
            feature_names.extend(oh_encoder.get_feature_names_out(cat_cols))

        processed_df = pd.DataFrame(processed_array, columns=feature_names, index=df.index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    return processed_df
