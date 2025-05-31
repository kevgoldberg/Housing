#!/usr/bin/env python3
"""Benchmark several regression models on the housing dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Add src directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_evaluator import benchmark_regressors


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regression models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/V2/train_advanced_cleaned.csv",
        help="Path to the processed training data",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="SalePrice",
        help="Name of the target column",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the results as CSV",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    results = benchmark_regressors(
        df, target=args.target, test_size=args.test_size
    )

    print("\nModel benchmark results (sorted by RMSE):")
    print(results.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
