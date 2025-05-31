#!/usr/bin/env python3
"""Simple command-line interface for running the data pipeline."""
import argparse
from pathlib import Path
import sys

# Add src directory to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

from data.pipeline import load_data, clean_data, save_data, create_visualizations


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")
    parser.add_argument("--input", required=True, help="Path to raw data CSV")
    parser.add_argument(
        "--output-data",
        default=str(project_root / "data/processed/cleaned_data.csv"),
        help="Output path for cleaned data",
    )
    parser.add_argument(
        "--viz-dir",
        default=str(project_root / "data/processed/visualizations"),
        help="Directory to save visualizations",
    )
    args = parser.parse_args()

    df = load_data(args.input)
    df_clean = clean_data(df)
    save_data(df_clean, args.output_data)
    create_visualizations(df_clean, args.viz_dir)

    print(f"Clean data saved to {args.output_data}")
    print(f"Visualizations stored in {args.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
