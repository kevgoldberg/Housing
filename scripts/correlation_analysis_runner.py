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
import pandas as pd
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from analysis.correlation_analyzer import HousingCorrelationAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Run correlation analysis on housing data')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the housing data CSV file')
    parser.add_argument('--target', type=str, default='SalePrice',
                       help='Target variable name (default: SalePrice)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Correlation threshold for flagging high correlations (default: 0.7)')
    parser.add_argument('--vif-threshold', type=float, default=5.0,
                       help='VIF threshold for multicollinearity detection (default: 5.0)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save correlation plots to files')
    parser.add_argument('--plots-dir', type=str, default='correlation_plots',
                       help='Directory to save plots (default: correlation_plots)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis (correlation matrix and high correlations only)')
    
    args = parser.parse_args()
    
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
        
    else:
        # Full analysis
        results = analyzer.generate_comprehensive_report(
            target_var=target_var,
            correlation_threshold=args.threshold,
            vif_threshold=args.vif_threshold,
            save_plots=args.save_plots,
            plots_dir=args.plots_dir
        )
        
        if args.save_plots:
            print(f"\nPlots saved to: {args.plots_dir}")
    
    print("\nAnalysis completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
