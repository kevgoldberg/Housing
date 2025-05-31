# Correlation Analysis System

This directory contains a comprehensive correlation analysis system designed specifically for housing data analysis before model development.

## Files Overview

### Core Components

- **`src/correlation_analyzer.py`** - Main correlation analysis module with the `HousingCorrelationAnalyzer` class
- **`notebooks/03_correlation_analysis.ipynb`** - Interactive Jupyter notebook for comprehensive analysis
- **`scripts/correlation_analysis_runner.py`** - Command-line script for quick analysis
- **`config/correlation_analysis_config.yaml`** - Configuration file for analysis settings

### Analysis Capabilities

#### 1. Correlation Matrix Analysis
- Pearson, Spearman, and Kendall correlations
- Interactive heatmap visualizations
- Correlation strength distribution analysis

#### 2. High Correlation Detection
- Configurable correlation thresholds
- Identification of potentially redundant features
- Relationship type classification (positive/negative)

#### 3. Multicollinearity Analysis
- Variance Inflation Factor (VIF) calculation
- Automatic flagging of problematic variables
- Interpretation guidelines for VIF scores

#### 4. Categorical Variable Associations
- Cramér's V statistics for categorical relationships
- Chi-square based association strength
- Cross-tabulation analysis

#### 5. Mixed Variable Analysis
- ANOVA F-statistics for categorical vs numeric relationships
- Effect size calculation (eta-squared)
- Statistical significance testing

#### 6. Feature Importance Analysis
- Correlation-based importance ranking
- Mutual information scores
- Support for both numeric and categorical features

## Usage Instructions

### 1. Interactive Analysis (Recommended)

```bash
# Open the Jupyter notebook
jupyter notebook notebooks/03_correlation_analysis.ipynb
```

The notebook provides:
- Step-by-step guided analysis
- Interactive visualizations
- Detailed explanations
- Comprehensive reporting

### 2. Command-Line Analysis

```bash
# Basic analysis
python scripts/correlation_analysis_runner.py --data data/processed/V1/train_cleaned.csv

# With custom target variable and threshold
python scripts/correlation_analysis_runner.py \
    --data data/processed/V1/train_cleaned.csv \
    --target SalePrice \
    --threshold 0.6 \
    --save-plots

# Quick analysis (correlation matrix only)
python scripts/correlation_analysis_runner.py \
    --data data/processed/V1/train_cleaned.csv \
    --quick
```

### 3. Programmatic Usage

```python
from src.correlation_analyzer import HousingCorrelationAnalyzer
import pandas as pd

# Load your data
df = pd.read_csv('data/processed/V1/train_cleaned.csv')

# Initialize analyzer
analyzer = HousingCorrelationAnalyzer(df)

# Generate comprehensive report
results = analyzer.generate_comprehensive_report(
    target_var='SalePrice',
    correlation_threshold=0.7,
    save_plots=True
)

# Individual analyses
corr_matrix = analyzer.correlation_matrix_analysis()
high_corr = analyzer.find_high_correlations(threshold=0.7)
vif_results = analyzer.multicollinearity_analysis()
```

## Analysis Outputs

### 1. Correlation Insights
- **High Correlations**: Variable pairs with strong linear relationships
- **Multicollinearity**: Variables that may cause modeling issues
- **Feature Redundancy**: Opportunities for dimensionality reduction

### 2. Feature Selection Guidance
- **Importance Rankings**: Variables ranked by predictive potential
- **Categorical Impact**: Effect sizes for categorical variables
- **Statistical Significance**: P-values for relationship tests

### 3. Modeling Recommendations
- **Model Type Suggestions**: Based on correlation structure
- **Preprocessing Steps**: Data transformation recommendations
- **Feature Engineering**: Opportunities for feature combination

### 4. Visualizations
- Correlation heatmaps
- Feature importance rankings
- VIF score distributions
- Effect size comparisons

## Configuration

Edit `config/correlation_analysis_config.yaml` to customize:

- **Analysis thresholds** (correlation, VIF, effect size)
- **Visualization settings** (colors, figure sizes, formats)
- **Feature selection criteria**
- **Statistical test parameters**

## Dependencies

Required Python packages:
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
statsmodels>=0.12.0  # Optional, for VIF calculation
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

## Interpretation Guidelines

### Correlation Strength
- **0.0 - 0.3**: Very weak
- **0.3 - 0.5**: Weak  
- **0.5 - 0.7**: Moderate
- **0.7 - 0.9**: Strong
- **0.9 - 1.0**: Very strong

### VIF (Variance Inflation Factor)
- **< 5**: Low multicollinearity
- **5 - 10**: Moderate multicollinearity  
- **> 10**: High multicollinearity (problematic)

### Effect Size (Eta-squared)
- **< 0.01**: Very small effect
- **0.01 - 0.06**: Small effect
- **0.06 - 0.14**: Medium effect
- **> 0.14**: Large effect

### Cramér's V (Categorical Association)
- **0.0 - 0.1**: Negligible association
- **0.1 - 0.3**: Weak association
- **0.3 - 0.5**: Moderate association
- **> 0.5**: Strong association

## Best Practices

1. **Always run correlation analysis before modeling** to understand feature relationships
2. **Use multiple correlation methods** (Pearson, Spearman) for robust insights
3. **Address multicollinearity** through feature selection or regularization
4. **Consider both statistical and practical significance** when interpreting results
5. **Document findings** for reproducible model development
6. **Validate insights** with domain knowledge

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**
   - Reduce figure sizes in config
   - Use correlation subsets
   - Enable chunking in advanced settings

2. **Missing statsmodels for VIF**
   - Install: `pip install statsmodels`
   - VIF analysis will be skipped if unavailable

3. **Categorical encoding errors**
   - Check for missing values in categorical columns
   - Verify category counts meet minimum thresholds

4. **Visualization issues**
   - Ensure matplotlib backend is properly configured
   - Check figure size settings in config

## Next Steps

After correlation analysis:

1. **Feature Engineering**: Create new features based on strong relationships
2. **Feature Selection**: Remove redundant or low-importance features  
3. **Model Selection**: Choose algorithms appropriate for correlation structure
4. **Preprocessing**: Apply transformations based on analysis findings
5. **Validation Strategy**: Design cross-validation considering feature relationships

---

*This correlation analysis system provides the foundation for data-driven model development in the housing price prediction project.*
