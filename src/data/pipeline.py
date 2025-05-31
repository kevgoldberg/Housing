import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simple cleaning: drop rows with missing values."""
    return df.dropna(axis=0, how="any").reset_index(drop=True)


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to CSV, creating directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def create_visualizations(df: pd.DataFrame, output_dir: str, target: str = "SalePrice") -> None:
    """Create simple visualizations and save them to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if target in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[target].dropna(), kde=True)
        plt.title(f"{target} Distribution")
        plt.tight_layout()
        plt.savefig(output_path / f"{target}_distribution.png")
        plt.close()

    numeric_cols = df.select_dtypes(include=[float, int]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="RdBu_r")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(output_path / "correlation_matrix.png")
        plt.close()

