# %% [markdown]
# # Housing Data Correlation Analysis
# 
# **Comprehensive correlation and association analysis before model development**
# 
# ---
# 
# ## Objective
# This notebook performs a thorough correlation analysis of the housing dataset to:
# - Identify relationships between variables
# - Detect multicollinearity issues
# - Analyze categorical associations
# - Rank feature importance
# - Provide modeling recommendations
# 
# ## Contents
# 1. [Data Loading and Setup](#1-data-loading-and-setup)
# 2. [Exploratory Data Overview](#2-exploratory-data-overview)
# 3. [Correlation Matrix Analysis](#3-correlation-matrix-analysis)
# 4. [High Correlation Detection](#4-high-correlation-detection)
# 5. [Multicollinearity Analysis](#5-multicollinearity-analysis)
# 6. [Categorical Variable Associations](#6-categorical-variable-associations)
# 7. [Mixed Variable Analysis](#7-mixed-variable-analysis)
# 8. [Feature Importance Analysis](#8-feature-importance-analysis)
# 9. [Comprehensive Report](#9-comprehensive-report)
# 10. [Modeling Recommendations](#10-modeling-recommendations)
# 
# ---

# %% [markdown]
# ## 1. Data Loading and Setup

# %%
# Import required libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src directory to path for imports
project_root = Path().resolve().parent
sys.path.append(str(project_root / 'src'))

# Import custom correlation analyzer
from correlation_analyzer import HousingCorrelationAnalyzer

# Set visualization style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("Setup completed successfully!")
print(f"Project root: {project_root}")

# %%
# Load the cleaned housing data
data_path = project_root / 'data' / 'processed' / 'V1' / 'train_cleaned.csv'

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

print(f"Data loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# %% [markdown]
# ## 2. Exploratory Data Overview

# %%
# Basic data information
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"\nColumn Types:")
print(df.dtypes.value_counts())

print(f"\nMissing Values Summary:")
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
if len(missing_summary) > 0:
    print(missing_summary.head(10))
else:
    print("No missing values found!")

# %%
# Identify numeric and categorical variables
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove ID columns from numeric if present
id_cols = [col for col in numeric_cols if 'id' in col.lower()]
numeric_cols = [col for col in numeric_cols if col not in id_cols]

print(f"Numeric variables ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
print(f"Categorical variables ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols) > 10 else ''}")
if id_cols:
    print(f"ID columns excluded: {id_cols}")

# %%
# Quick statistical summary for numeric variables
print("=== NUMERIC VARIABLES SUMMARY ===")
numeric_summary = df[numeric_cols].describe()
print(numeric_summary.round(2))

# %%
# Target variable analysis (assuming SalePrice is the target)
target_var = 'SalePrice' if 'SalePrice' in df.columns else None

if target_var:
    print(f"=== TARGET VARIABLE ANALYSIS: {target_var} ===")
    print(f"Statistics:")
    print(df[target_var].describe())
    
    # Plot target distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    axes[0].hist(df[target_var], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'{target_var} Distribution')
    axes[0].set_xlabel(target_var)
    axes[0].set_ylabel('Frequency')
    
    # Log-transformed histogram
    axes[1].hist(np.log1p(df[target_var]), bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_title(f'Log({target_var}) Distribution')
    axes[1].set_xlabel(f'Log({target_var})')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Skewness analysis
    from scipy.stats import skew
    original_skew = skew(df[target_var])
    log_skew = skew(np.log1p(df[target_var]))
    print(f"\nSkewness Analysis:")
    print(f"Original {target_var} skewness: {original_skew:.3f}")
    print(f"Log-transformed skewness: {log_skew:.3f}")
else:
    print("No target variable identified. Please specify target variable for focused analysis.")

# %% [markdown]
# ## 3. Correlation Matrix Analysis

# %%
# Initialize the correlation analyzer
analyzer = HousingCorrelationAnalyzer(df)

print("Correlation analyzer initialized successfully!")

# %%
# Generate correlation matrix
print("Generating correlation matrix...")
correlation_matrix = analyzer.correlation_matrix_analysis(method='pearson', figsize=(16, 14))

# %%
# Show correlation matrix statistics
if not correlation_matrix.empty:
    print("=== CORRELATION MATRIX STATISTICS ===")
    
    # Get upper triangle correlations (excluding diagonal)
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    correlations = upper_tri.stack().dropna()
    
    print(f"Total correlation pairs: {len(correlations)}")
    print(f"Mean absolute correlation: {abs(correlations).mean():.3f}")
    print(f"Median absolute correlation: {abs(correlations).median():.3f}")
    print(f"Max correlation: {correlations.max():.3f}")
    print(f"Min correlation: {correlations.min():.3f}")
    
    # Distribution of correlation strengths
    abs_corr = abs(correlations)
    print(f"\nCorrelation Strength Distribution:")
    print(f"Very weak (< 0.3): {(abs_corr < 0.3).sum()} ({(abs_corr < 0.3).mean()*100:.1f}%)")
    print(f"Weak (0.3-0.5): {((abs_corr >= 0.3) & (abs_corr < 0.5)).sum()} ({((abs_corr >= 0.3) & (abs_corr < 0.5)).mean()*100:.1f}%)")
    print(f"Moderate (0.5-0.7): {((abs_corr >= 0.5) & (abs_corr < 0.7)).sum()} ({((abs_corr >= 0.5) & (abs_corr < 0.7)).mean()*100:.1f}%)")
    print(f"Strong (0.7-0.9): {((abs_corr >= 0.7) & (abs_corr < 0.9)).sum()} ({((abs_corr >= 0.7) & (abs_corr < 0.9)).mean()*100:.1f}%)")
    print(f"Very strong (≥ 0.9): {(abs_corr >= 0.9).sum()} ({(abs_corr >= 0.9).mean()*100:.1f}%)")

# %% [markdown]
# ## 4. High Correlation Detection

# %%
# Find high correlations
print("Analyzing high correlations...")
high_correlations = analyzer.find_high_correlations(threshold=0.7, target_var=target_var)

if not high_correlations.empty:
    print(f"\n=== HIGH CORRELATIONS (|r| ≥ 0.7) ===")
    print(f"Found {len(high_correlations)} highly correlated pairs:")
    print(high_correlations.to_string(index=False))
    
    # Visualization of high correlations
    if len(high_correlations) > 0:
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(high_correlations)), 
                       high_correlations['Correlation'],
                       color=['red' if x < 0 else 'blue' for x in high_correlations['Correlation']])
        
        # Create labels
        labels = [f"{row['Variable_1']} vs {row['Variable_2']}" 
                 for _, row in high_correlations.iterrows()]
        
        plt.yticks(range(len(high_correlations)), labels)
        plt.xlabel('Correlation Coefficient')
        plt.title('High Correlations (|r| ≥ 0.7)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold')
        plt.axvline(x=-0.7, color='red', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
else:
    print("No high correlations found with threshold 0.7")

# %%
# Try lower threshold if no high correlations found
if high_correlations.empty:
    print("Trying lower threshold (0.5)...")
    moderate_correlations = analyzer.find_high_correlations(threshold=0.5, target_var=target_var)
    
    if not moderate_correlations.empty:
        print(f"\n=== MODERATE CORRELATIONS (|r| ≥ 0.5) ===")
        print(f"Found {len(moderate_correlations)} moderately correlated pairs:")
        print(moderate_correlations.head(10).to_string(index=False))
    else:
        print("No correlations found even with threshold 0.5")

# %% [markdown]
# ## 5. Multicollinearity Analysis

# %%
# Check for multicollinearity using VIF
print("Analyzing multicollinearity (VIF)...")
vif_results = analyzer.multicollinearity_analysis(threshold=5.0)

if not vif_results.empty:
    print(f"\n=== VARIANCE INFLATION FACTOR (VIF) ANALYSIS ===")
    print(vif_results.to_string(index=False))
    
    # Count problematic variables
    high_vif = vif_results[vif_results['High_Multicollinearity']]
    if not high_vif.empty:
        print(f"\n⚠️ WARNING: {len(high_vif)} variables have VIF > 5.0")
        print("These may cause multicollinearity issues:")
        print(high_vif[['Variable', 'VIF', 'VIF_Interpretation']].to_string(index=False))
    else:
        print("\n✅ No multicollinearity concerns detected (all VIF < 5.0)")
    
    # Visualize VIF scores
    if len(vif_results) > 1:
        plt.figure(figsize=(12, 8))
        colors = ['red' if x else 'green' for x in vif_results['High_Multicollinearity']]
        
        bars = plt.barh(range(len(vif_results)), vif_results['VIF'], color=colors, alpha=0.7)
        plt.yticks(range(len(vif_results)), vif_results['Variable'])
        plt.xlabel('VIF Score')
        plt.title('Variance Inflation Factor (VIF) by Variable')
        plt.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Threshold (VIF=5)')
        plt.axvline(x=10, color='darkred', linestyle='--', alpha=0.7, label='High concern (VIF=10)')
        plt.legend()
        
        # Add value labels on bars
        for i, (bar, vif_val) in enumerate(zip(bars, vif_results['VIF'])):
            plt.text(vif_val + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{vif_val:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
else:
    print("VIF analysis could not be performed (insufficient data or statsmodels not available)")

# %% [markdown]
# ## 6. Categorical Variable Associations

# %%
# Analyze categorical variable associations
if len(categorical_cols) > 1:
    print("Analyzing categorical variable associations...")
    categorical_associations = analyzer.categorical_associations_analysis()
    
    if not categorical_associations.empty:
        print(f"\n=== CATEGORICAL ASSOCIATIONS (Cramér's V) ===")
        print(f"Analyzed {len(categorical_associations)} categorical variable pairs")
        print("\nTop 15 associations:")
        print(categorical_associations.head(15).to_string())
        
        # Strong associations
        strong_associations = categorical_associations[categorical_associations > 0.5]
        if len(strong_associations) > 0:
            print(f"\nStrong associations (Cramér's V > 0.5): {len(strong_associations)}")
            print(strong_associations.to_string())
        else:
            print("\nNo strong categorical associations found (all Cramér's V ≤ 0.5)")
        
        # Visualize top associations
        if len(categorical_associations) > 0:
            top_10 = categorical_associations.head(10)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_10)), top_10.values)
            plt.yticks(range(len(top_10)), top_10.index)
            plt.xlabel("Cramér's V")
            plt.title("Top 10 Categorical Variable Associations")
            plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Strong association (0.5)')
            plt.legend()
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top_10.values)):
                plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()
    else:
        print("No categorical associations could be calculated")
else:
    print("Insufficient categorical variables for association analysis")

# %% [markdown]
# ## 7. Mixed Variable Analysis

# %%
# Analyze relationships between categorical and numeric variables
if target_var and len(categorical_cols) > 0:
    print("Analyzing categorical vs numeric relationships...")
    
    # Test top categorical variables against target
    mixed_results = {}
    
    for cat_var in categorical_cols[:8]:  # Analyze top 8 categorical variables
        try:
            result = analyzer.mixed_variable_analysis(cat_var, [target_var])
            if not result.empty:
                mixed_results[cat_var] = result.iloc[0]
        except Exception as e:
            print(f"Could not analyze {cat_var}: {e}")
    
    if mixed_results:
        mixed_df = pd.DataFrame(mixed_results).T
        mixed_df = mixed_df.sort_values('eta_squared', ascending=False)
        
        print(f"\n=== CATEGORICAL vs {target_var} ANALYSIS ===")
        print("Effect sizes (η²) and significance:")
        display_cols = ['F_statistic', 'p_value', 'eta_squared', 'effect_size', 'significant_005']
        print(mixed_df[display_cols].round(4).to_string())
        
        # Visualize effect sizes
        plt.figure(figsize=(12, 6))
        colors = ['red' if x else 'blue' for x in mixed_df['significant_005']]
        bars = plt.bar(range(len(mixed_df)), mixed_df['eta_squared'], color=colors, alpha=0.7)
        plt.xticks(range(len(mixed_df)), mixed_df.index, rotation=45, ha='right')
        plt.ylabel('Eta-squared (η²)')
        plt.title(f'Effect Sizes: Categorical Variables vs {target_var}')
        
        # Add significance indicators
        for i, (bar, significant) in enumerate(zip(bars, mixed_df['significant_005'])):
            if significant:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        '*', ha='center', va='bottom', fontsize=16, color='red')
        
        plt.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        plt.axhline(y=0.06, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        plt.axhline(y=0.14, color='red', linestyle='--', alpha=0.5, label='Large effect')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Show significant relationships
        significant_vars = mixed_df[mixed_df['significant_005']]
        if not significant_vars.empty:
            print(f"\nSignificant relationships with {target_var} (p < 0.05): {len(significant_vars)}")
            print(significant_vars[['eta_squared', 'effect_size']].to_string())
else:
    print("Mixed variable analysis skipped (no target variable or categorical variables)")

# %% [markdown]
# ## 8. Feature Importance Analysis

# %%
# Feature importance analysis
if target_var:
    print("Analyzing feature importance...")
    
    # Try both correlation and mutual information methods
    methods = ['correlation', 'mutual_info']
    importance_results = {}
    
    for method in methods:
        try:
            importance = analyzer.feature_importance_analysis(target_var, method=method)
            if not importance.empty:
                importance_results[method] = importance
                print(f"\n=== FEATURE IMPORTANCE: {method.upper()} METHOD ===")
                print("Top 15 features:")
                print(importance.head(15).to_string(index=False))
        except Exception as e:
            print(f"Could not calculate {method} importance: {e}")
    
    # Visualize feature importance
    if importance_results:
        for method, importance in importance_results.items():
            if len(importance) > 1:
                top_features = importance.head(15)
                
                plt.figure(figsize=(12, 8))
                colors = ['orange' if ft == 'Categorical' else 'blue' 
                         for ft in top_features.get('Feature_Type', ['Numeric'] * len(top_features))]
                
                bars = plt.barh(range(len(top_features)), 
                               top_features.iloc[:, 1], color=colors, alpha=0.7)
                plt.yticks(range(len(top_features)), top_features['Feature'])
                plt.xlabel(f'Importance Score ({method})')
                plt.title(f'Top 15 Features by {method.title()} Importance')
                plt.gca().invert_yaxis()
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, top_features.iloc[:, 1])):
                    plt.text(val + max(top_features.iloc[:, 1]) * 0.01, 
                            bar.get_y() + bar.get_height()/2, 
                            f'{val:.3f}', va='center', fontsize=9)
                
                if 'Feature_Type' in top_features.columns:
                    # Add legend for feature types
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Numeric'),
                                     Patch(facecolor='orange', alpha=0.7, label='Categorical')]
                    plt.legend(handles=legend_elements)
                
                plt.tight_layout()
                plt.show()
else:
    print("Feature importance analysis skipped (no target variable specified)")

# %% [markdown]
# ## 9. Comprehensive Report

# %%
# Generate comprehensive report
print("Generating comprehensive correlation analysis report...")

# Create plots directory
plots_dir = project_root / 'notebooks' / 'correlation_analysis_plots'
plots_dir.mkdir(exist_ok=True)

# Generate full report
report_results = analyzer.generate_comprehensive_report(
    target_var=target_var,
    correlation_threshold=0.7,
    vif_threshold=5.0,
    save_plots=True,
    plots_dir=str(plots_dir)
)

print(f"\nPlots saved to: {plots_dir}")

# %% [markdown]
# ## 10. Modeling Recommendations

# %%
# Extract and display detailed modeling recommendations
print("=" * 60)
print("DETAILED MODELING RECOMMENDATIONS")
print("=" * 60)

recommendations = []

# 1. Feature Selection Recommendations
print("\n1. FEATURE SELECTION RECOMMENDATIONS")
print("-" * 40)

if 'feature_importance' in report_results and not report_results['feature_importance'].empty:
    importance_df = report_results['feature_importance']
    top_features = importance_df.head(20)['Feature'].tolist()
    print(f"• Focus on top 20 features with highest importance:")
    print(f"  {', '.join(top_features[:10])}")
    if len(top_features) > 10:
        print(f"  {', '.join(top_features[10:])}")
    recommendations.append("Use feature importance ranking for initial feature selection")

# 2. Multicollinearity Handling
print("\n2. MULTICOLLINEARITY HANDLING")
print("-" * 40)

if 'vif_analysis' in report_results and not report_results['vif_analysis'].empty:
    high_vif_vars = report_results['vif_analysis'][report_results['vif_analysis']['High_Multicollinearity']]
    if not high_vif_vars.empty:
        print(f"• Remove or combine these {len(high_vif_vars)} high-VIF variables:")
        for _, row in high_vif_vars.iterrows():
            print(f"  - {row['Variable']} (VIF: {row['VIF']:.2f})")
        recommendations.append("Address multicollinearity before modeling")
    else:
        print("• ✅ No multicollinearity issues detected")
        recommendations.append("Multicollinearity is not a concern for this dataset")

# 3. Correlation-based Recommendations
print("\n3. CORRELATION-BASED RECOMMENDATIONS")
print("-" * 40)

if 'high_correlations' in report_results and not report_results['high_correlations'].empty:
    high_corr_df = report_results['high_correlations']
    print(f"• Consider feature pairs with high correlation ({len(high_corr_df)} pairs):")
    for _, row in high_corr_df.head(5).iterrows():
        print(f"  - {row['Variable_1']} vs {row['Variable_2']} (r={row['Correlation']:.3f})")
    recommendations.append("Use regularization or feature selection to handle correlated features")
else:
    print("• ✅ No problematic high correlations found")
    recommendations.append("Feature correlations are manageable")

# 4. Target Variable Recommendations
if target_var:
    print(f"\n4. TARGET VARIABLE ({target_var}) RECOMMENDATIONS")
    print("-" * 40)
    
    # Check skewness
    if target_var in df.columns:
        from scipy.stats import skew
        target_skew = skew(df[target_var])
        if abs(target_skew) > 1:
            print(f"• Target variable is skewed (skewness: {target_skew:.3f})")
            print(f"• Consider log transformation: log({target_var})")
            recommendations.append("Apply log transformation to target variable")
        else:
            print(f"• Target variable has acceptable skewness ({target_skew:.3f})")

# 5. Model Type Recommendations
print("\n5. RECOMMENDED MODEL TYPES")
print("-" * 40)

model_recommendations = []

# Based on correlation structure
if 'correlation_matrix' in report_results and not report_results['correlation_matrix'].empty:
    corr_matrix = report_results['correlation_matrix']
    mean_abs_corr = np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).mean()
    
    if mean_abs_corr > 0.3:
        model_recommendations.extend([
            "Ridge Regression (handles multicollinearity)",
            "Elastic Net (feature selection + multicollinearity)"
        ])
    else:
        model_recommendations.append("Linear Regression (low correlation between features)")

# Based on feature count
num_features = len(numeric_cols) + len(categorical_cols)
if num_features > 50:
    model_recommendations.extend([
        "Lasso Regression (automatic feature selection)",
        "Random Forest (handles many features well)"
    ])
elif num_features > 20:
    model_recommendations.extend([
        "Random Forest",
        "Gradient Boosting"
    ])

# Based on categorical variables
if len(categorical_cols) > 5:
    model_recommendations.extend([
        "Random Forest (handles categorical features)",
        "XGBoost (with proper encoding)"
    ])

print(f"• Recommended models based on data characteristics:")
for model in set(model_recommendations):
    print(f"  - {model}")

# 6. Data Preprocessing Recommendations
print("\n6. DATA PREPROCESSING RECOMMENDATIONS")
print("-" * 40)

preprocessing_steps = [
    "Standard scaling for numeric features (especially for linear models)",
    "One-hot encoding for categorical features with few categories",
    "Target encoding for high-cardinality categorical features"
]

if target_var and abs(skew(df[target_var])) > 1:
    preprocessing_steps.insert(0, f"Log transformation for {target_var}")

if 'high_correlations' in report_results and not report_results['high_correlations'].empty:
    preprocessing_steps.append("PCA or feature selection for correlated features")

for step in preprocessing_steps:
    print(f"• {step}")

# 7. Cross-validation Strategy
print("\n7. CROSS-VALIDATION STRATEGY")
print("-" * 40)
print(f"• Use 5-fold cross-validation for model selection")
print(f"• Stratified sampling if target has skewed distribution")
print(f"• Time-based split if temporal features are present")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE - READY FOR MODELING")
print("=" * 60)

# %% [markdown]
# ## Summary
# 
# This comprehensive correlation analysis has provided insights into:
# 
# 1. **Data Structure**: Understanding of numeric vs categorical variables
# 2. **Correlation Patterns**: Identification of variable relationships
# 3. **Multicollinearity**: Detection of problematic feature combinations
# 4. **Feature Importance**: Ranking of variables by predictive potential
# 5. **Modeling Strategy**: Data-driven recommendations for model selection
# 
# **Next Steps:**
# - Implement recommended preprocessing steps
# - Start with suggested model types
# - Use identified important features for initial models
# - Address any multicollinearity concerns
# - Consider target variable transformations if needed
# 
# The analysis results and plots have been saved for reference during model development.
