#!/usr/bin/env python3
"""Command-line interface for simple preprocessing of the Kaggle housing data."""
import argparse
import sys
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root / "src"))

from data_cleaner import simple_preprocess, advanced_preprocess
from data_analyzer import summarize_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data preprocessing on housing data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/train.csv",
        help="Path to the raw input CSV (default: data/raw/train.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/V1/train_cleaned.csv",
        help="Destination for the cleaned CSV (default: data/processed/V1/train_cleaned.csv)",
    )
    parser.add_argument(
        "--method",
        choices=["simple", "advanced"],
        default="simple",
        help="Preprocessing strategy to use (simple or advanced)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Display a summary of the dataset before preprocessing",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.input}")

    if args.analyze:
        try:
            summary = summarize_csv(args.input)
            print("\nDataset summary:\n" + summary.to_string(index=False))
        except Exception as exc:
            print(f"Failed to analyze dataset: {exc}")

    if args.method == "advanced":
        df = advanced_preprocess(args.input, args.output)
    else:
        df = simple_preprocess(args.input, args.output)
    print(f"Saved cleaned data to {args.output} (shape: {df.shape})")


if __name__ == "__main__":
    main()
