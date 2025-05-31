#!/usr/bin/env python3
"""
Quick Correlation Analysis Script

This script provides a command-line interface for running correlation analysis
on housing data. It can be used for quick checks during model development.

Usage:
    python correlation_analysis_runner.py --data path/to/data.csv --target SalePrice
    python correlation_analysis_runner.py --data path/to/data.csv --target SalePrice --threshold 0.5
    python correlation_analysis_runner.py --help

Author: Housing Analysis Project
Date: May 31, 2025
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from correlation_analyzer import HousingCorrelationAnalyzer


def load_config(path: Path) -> dict:
    """Load YAML configuration file."""
    try:
        with path.open("r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Config file not found: {path}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Run correlation analysis on housing data')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML configuration file')
    parser.add_argument('--data', type=str, required=False,
                       help='Path to the housing data CSV file')
    parser.add_argument('--target', type=str, default=None,
                        help='Target variable name')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Correlation threshold for flagging high correlations')
    parser.add_argument('--vif-threshold', type=float, default=None,
                        help='VIF threshold for multicollinearity detection')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save correlation plots to files')
    parser.add_argument('--plots-dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis (correlation matrix and high correlations only)')
    parser.add_argument('--output-report', type=str, default=None,
                        help='Optional path to save analysis results as JSON')
    
    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(Path(args.config))

    # Apply config defaults
    if args.data is None:
        args.data = config.get('data', {}).get('train_path')
    if args.target is None:
        args.target = config.get('analysis', {}).get('target_variable', 'SalePrice')
    if args.threshold is None:
        args.threshold = config.get('analysis', {}).get('correlation_threshold', 0.7)
    if args.vif_threshold is None:
        args.vif_threshold = config.get('analysis', {}).get('vif_threshold', 5.0)
    if not args.save_plots:
        args.save_plots = config.get('visualization', {}).get('save_plots', False)
    if args.plots_dir is None:
        args.plots_dir = config.get('visualization', {}).get('plots_dir', 'correlation_plots')
    if args.output_report is None:
        args.output_report = config.get('reporting', {}).get('output_report')

    if not args.data:
        parser.error('Data path must be provided via --data or config file.')

    # Load data
    try:
        print(f"Loading data from: {args.data}")
        df = pd.read_csv(args.data)
        print(f"Data loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Check if target variable exists
    if args.target not in df.columns:
        print(f"Warning: Target variable '{args.target}' not found in dataset")
        print(f"Available columns: {', '.join(df.columns.tolist()[:10])}...")
        target_var = None
    else:
        target_var = args.target
    
    # Initialize analyzer
    analyzer = HousingCorrelationAnalyzer(df)
    
    if args.quick:
        # Quick analysis
        print("\n" + "="*50)
        print("QUICK CORRELATION ANALYSIS")
        print("="*50)
        
        # Correlation matrix
        corr_matrix = analyzer.correlation_matrix_analysis()

        # High correlations
        high_corr = analyzer.find_high_correlations(threshold=args.threshold, target_var=target_var)
        if not high_corr.empty:
            print(f"\nHigh correlations (|r| >= {args.threshold}):")
            print(high_corr.to_string(index=False))
        else:
            print(f"\nNo high correlations found with threshold {args.threshold}")

        if args.output_report:
            analyzer.save_report({
                'correlation_matrix': corr_matrix,
                'high_correlations': high_corr
            }, args.output_report)

    else:
        # Full analysis
        results = analyzer.generate_comprehensive_report(
            target_var=target_var,
            correlation_threshold=args.threshold,
            vif_threshold=args.vif_threshold,
            save_plots=args.save_plots,
            plots_dir=args.plots_dir,
            output_path=args.output_report
        )
        
        if args.save_plots:
            print(f"\nPlots saved to: {args.plots_dir}")
    
    print("\nAnalysis completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
