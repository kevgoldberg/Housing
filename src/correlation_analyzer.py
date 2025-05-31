"""
Comprehensive Correlation Analysis Module for Housing Data

This module provides tools for analyzing correlations, associations, and relationships
between variables in the housing dataset before model creation.

Author: Housing Analysis Project
Date: May 31, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
from typing import Optional, List, Dict, Tuple, Union
import itertools
import json
from pathlib import Path

warnings.filterwarnings('ignore')

class HousingCorrelationAnalyzer:
    """
    A comprehensive analyzer for correlation and association analysis
    specifically designed for housing data analysis.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a housing dataset.
        
        Args:
            df (pd.DataFrame): The housing dataset to analyze
        """
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.original_shape = df.shape
        
        # Remove ID columns if present
        id_cols = [col for col in self.numeric_cols if 'id' in col.lower()]
        self.numeric_cols = [col for col in self.numeric_cols if col not in id_cols]
        
        print(f"Dataset loaded: {self.original_shape[0]} rows, {self.original_shape[1]} columns")
        print(f"Numeric variables: {len(self.numeric_cols)}")
        print(f"Categorical variables: {len(self.categorical_cols)}")
    
    def correlation_matrix_analysis(self, 
                                  method: str = 'pearson', 
                                  figsize: Tuple[int, int] = (14, 12),
                                  save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate and visualize correlation matrix for numeric variables.
        
        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            figsize (Tuple[int, int]): Figure size for the heatmap
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if len(self.numeric_cols) < 2:
            print("Warning: Less than 2 numeric columns available for correlation analysis")
            return pd.DataFrame()
        
        corr_matrix = self.df[self.numeric_cols].corr(method=method)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0, 
                   square=True, 
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   annot_kws={'size': 8})
        
        plt.title(f'{method.capitalize()} Correlation Matrix - Housing Data', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to: {save_path}")
        
        plt.show()
        
        return corr_matrix
    
    def find_high_correlations(self, 
                             threshold: float = 0.7, 
                             target_var: Optional[str] = None) -> pd.DataFrame:
        """
        Identify highly correlated variable pairs.
        
        Args:
            threshold (float): Correlation threshold for flagging high correlations
            target_var (Optional[str]): Target variable to focus analysis on
            
        Returns:
            pd.DataFrame: DataFrame of high correlation pairs
        """
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Get upper triangle of correlation matrix
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs with high correlation
        high_corr_pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                corr_val = upper_tri.loc[idx, col]
                if pd.notna(corr_val) and abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'Variable_1': idx,
                        'Variable_2': col,
                        'Correlation': corr_val,
                        'Abs_Correlation': abs(corr_val),
                        'Relationship': 'Positive' if corr_val > 0 else 'Negative'
                    })
        
        high_corr_df = pd.DataFrame(high_corr_pairs)
        if not high_corr_df.empty:
            high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False)
        
        # Analyze target variable correlations if specified
        if target_var and target_var in self.numeric_cols:
            print(f"\n=== CORRELATIONS WITH TARGET VARIABLE: {target_var} ===")
            target_corr = corr_matrix[target_var].sort_values(key=abs, ascending=False)
            target_corr = target_corr.drop(target_var)  # Remove self-correlation
            
            print("\nTop 10 Strongest Correlations with Target:")
            print(target_corr.head(10).to_string())
            
            print(f"\nVariables with |correlation| > {threshold} with {target_var}:")
            strong_target_corr = target_corr[abs(target_corr) > threshold]
            if len(strong_target_corr) > 0:
                print(strong_target_corr.to_string())
            else:
                print("No variables meet the threshold criteria.")
        
        return high_corr_df
    
    def categorical_associations_analysis(self) -> pd.Series:
        """
        Analyze associations between categorical variables using Cramér's V.
        
        Returns:
            pd.Series: Cramér's V values for categorical variable pairs
        """
        def cramers_v(x: pd.Series, y: pd.Series) -> float:
            """Calculate Cramér's V statistic for categorical association"""
            try:
                confusion_matrix = pd.crosstab(x, y)
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                phi2 = chi2 / n
                r, k = confusion_matrix.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                if min((kcorr-1), (rcorr-1)) == 0:
                    return 0
                return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
            except:
                return np.nan
        
        if len(self.categorical_cols) < 2:
            print("Warning: Less than 2 categorical columns available for association analysis")
            return pd.Series()
        
        associations = {}
        cat_cols = self.categorical_cols.copy()
        
        print("Calculating Cramér's V for categorical variable pairs...")
        for i, var1 in enumerate(cat_cols):
            for var2 in cat_cols[i+1:]:
                v_stat = cramers_v(self.df[var1].dropna(), self.df[var2].dropna())
                associations[f"{var1} vs {var2}"] = v_stat
        
        return pd.Series(associations).sort_values(ascending=False)
    
    def mixed_variable_analysis(self, 
                              categorical_var: str, 
                              numeric_vars: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze relationships between categorical and numeric variables.
        
        Args:
            categorical_var (str): The categorical variable to analyze
            numeric_vars (Optional[List[str]]): List of numeric variables to test
            
        Returns:
            pd.DataFrame: Analysis results with F-statistics, p-values, and effect sizes
        """
        if categorical_var not in self.categorical_cols:
            raise ValueError(f"{categorical_var} is not a categorical variable in the dataset")
        
        if numeric_vars is None:
            numeric_vars = self.numeric_cols
        
        results = {}
        
        for num_var in numeric_vars:
            if num_var not in self.df.columns:
                continue
                
            # Prepare data
            clean_data = self.df[[categorical_var, num_var]].dropna()
            
            if clean_data.empty:
                continue
            
            # Group by categorical variable
            groups = [group[num_var].values for name, group in clean_data.groupby(categorical_var)]
            
            # Skip if any group is empty
            if any(len(group) == 0 for group in groups):
                continue
            
            try:
                # ANOVA F-statistic
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Calculate eta-squared (effect size)
                group_means = [np.mean(group) for group in groups]
                overall_mean = clean_data[num_var].mean()
                group_sizes = [len(group) for group in groups]
                
                ss_between = sum(size * (mean - overall_mean)**2 
                               for size, mean in zip(group_sizes, group_means))
                ss_total = ((clean_data[num_var] - overall_mean)**2).sum()
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                results[num_var] = {
                    'F_statistic': f_stat,
                    'p_value': p_value,
                    'eta_squared': eta_squared,
                    'significant_005': p_value < 0.05,
                    'significant_001': p_value < 0.01,
                    'effect_size': self._interpret_eta_squared(eta_squared)
                }
            except:
                results[num_var] = {
                    'F_statistic': np.nan,
                    'p_value': np.nan,
                    'eta_squared': np.nan,
                    'significant_005': False,
                    'significant_001': False,
                    'effect_size': 'Cannot calculate'
                }
        
        return pd.DataFrame(results).T.sort_values('eta_squared', ascending=False)
    
    def multicollinearity_analysis(self, threshold: float = 5.0) -> pd.DataFrame:
        """
        Check for multicollinearity using Variance Inflation Factor (VIF).
        
        Args:
            threshold (float): VIF threshold for flagging high multicollinearity
            
        Returns:
            pd.DataFrame: VIF analysis results
        """
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            print("Warning: statsmodels not available. Install with: pip install statsmodels")
            return pd.DataFrame()
        
        # Select only numeric columns with no missing values
        numeric_df = self.df[self.numeric_cols].dropna()
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            print("Warning: Insufficient data for VIF calculation")
            return pd.DataFrame()
        
        # Remove columns with zero variance
        numeric_df = numeric_df.loc[:, numeric_df.var() != 0]
        
        if numeric_df.shape[1] < 2:
            print("Warning: Insufficient columns with variance for VIF calculation")
            return pd.DataFrame()
        
        try:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = numeric_df.columns
            vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                              for i in range(numeric_df.shape[1])]
            
            vif_data = vif_data.sort_values('VIF', ascending=False)
            vif_data['High_Multicollinearity'] = vif_data['VIF'] > threshold
            vif_data['VIF_Interpretation'] = vif_data['VIF'].apply(self._interpret_vif)
            
            return vif_data
        except Exception as e:
            print(f"Error calculating VIF: {e}")
            return pd.DataFrame()
    
    def feature_importance_analysis(self, 
                                  target_var: str, 
                                  method: str = 'correlation') -> pd.DataFrame:
        """
        Calculate feature importance using correlation or mutual information.
        
        Args:
            target_var (str): Target variable for importance calculation
            method (str): Method to use ('correlation' or 'mutual_info')
            
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        if target_var not in self.df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in dataset")
        
        if method == 'mutual_info':
            return self._mutual_info_analysis(target_var)
        else:
            return self._correlation_importance_analysis(target_var)
    
    def _mutual_info_analysis(self, target_var: str) -> pd.DataFrame:
        """Calculate mutual information scores"""
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            print("Warning: scikit-learn not available. Using correlation method instead.")
            return self._correlation_importance_analysis(target_var)
        
        # Prepare features
        X = self.df.drop(columns=[target_var]).copy()
        y = self.df[target_var].copy()
        
        # Handle categorical variables
        categorical_features = []
        label_encoders = {}
        
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                categorical_features.append(True)
            else:
                categorical_features.append(False)
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if X_clean.empty:
            print("Warning: No complete cases available for mutual information analysis")
            return pd.DataFrame()
        
        try:
            if y_clean.dtype == 'object':
                # Classification
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y_clean.astype(str))
                mi_scores = mutual_info_classif(X_clean, y_encoded, 
                                              discrete_features=categorical_features,
                                              random_state=42)
            else:
                # Regression
                mi_scores = mutual_info_regression(X_clean, y_clean, 
                                                 discrete_features=categorical_features,
                                                 random_state=42)
            
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Mutual_Information': mi_scores,
                'Feature_Type': ['Categorical' if cat else 'Numeric' 
                               for cat in categorical_features]
            }).sort_values('Mutual_Information', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error in mutual information calculation: {e}")
            return self._correlation_importance_analysis(target_var)
    
    def _correlation_importance_analysis(self, target_var: str) -> pd.DataFrame:
        """Calculate importance using correlation for numeric variables"""
        if target_var not in self.numeric_cols:
            print(f"Warning: {target_var} is not numeric. Correlation analysis limited.")
            return pd.DataFrame()
        
        correlations = self.df[self.numeric_cols].corr()[target_var].abs()
        correlations = correlations.drop(target_var)  # Remove self-correlation
        
        importance_df = pd.DataFrame({
            'Feature': correlations.index,
            'Abs_Correlation': correlations.values,
            'Feature_Type': 'Numeric'
        }).sort_values('Abs_Correlation', ascending=False)
        
        return importance_df
    
    def generate_comprehensive_report(self,
                                    target_var: Optional[str] = None,
                                    correlation_threshold: float = 0.7,
                                    vif_threshold: float = 5.0,
                                    save_plots: bool = False,
                                    plots_dir: str = "plots",
                                    output_path: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive correlation analysis report.
        
        Args:
            target_var (Optional[str]): Target variable for focused analysis
            correlation_threshold (float): Threshold for flagging high correlations
            vif_threshold (float): Threshold for flagging high VIF
            save_plots (bool): Whether to save plots to files
            plots_dir (str): Directory to save plots
            output_path (Optional[str]): Optional path to save the report as JSON
            
        Returns:
            Dict: Dictionary containing all analysis results
        """
        print("=" * 60)
        print("COMPREHENSIVE HOUSING DATA CORRELATION ANALYSIS")
        print("=" * 60)
        
        print(f"\nDataset Overview:")
        print(f"- Shape: {self.original_shape[0]} rows × {self.original_shape[1]} columns")
        print(f"- Numeric variables: {len(self.numeric_cols)}")
        print(f"- Categorical variables: {len(self.categorical_cols)}")
        print(f"- Missing values: {self.df.isnull().sum().sum():,}")
        print(f"- Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        results = {}
        
        # 1. Correlation Matrix Analysis
        print(f"\n{'-'*50}")
        print("1. CORRELATION MATRIX ANALYSIS")
        print(f"{'-'*50}")
        
        if save_plots:
            import os
            os.makedirs(plots_dir, exist_ok=True)
            save_path = f"{plots_dir}/correlation_matrix.png"
        else:
            save_path = None
            
        corr_matrix = self.correlation_matrix_analysis(save_path=save_path)
        results['correlation_matrix'] = corr_matrix
        
        # 2. High Correlations
        print(f"\n{'-'*50}")
        print(f"2. HIGH CORRELATIONS (|r| ≥ {correlation_threshold})")
        print(f"{'-'*50}")
        
        high_corr = self.find_high_correlations(threshold=correlation_threshold, 
                                               target_var=target_var)
        if not high_corr.empty:
            print(f"\nFound {len(high_corr)} highly correlated pairs:")
            print(high_corr.to_string(index=False))
            
            # Identify potential multicollinearity issues
            variables_in_high_corr = set(high_corr['Variable_1']) | set(high_corr['Variable_2'])
            print(f"\nVariables involved in high correlations: {len(variables_in_high_corr)}")
            print(f"Consider reviewing: {', '.join(sorted(variables_in_high_corr))}")
        else:
            print("No high correlations found.")
        
        results['high_correlations'] = high_corr
        
        # 3. Multicollinearity Analysis
        print(f"\n{'-'*50}")
        print(f"3. MULTICOLLINEARITY ANALYSIS (VIF)")
        print(f"{'-'*50}")
        
        vif_results = self.multicollinearity_analysis(threshold=vif_threshold)
        if not vif_results.empty:
            print("\nVariance Inflation Factor (VIF) Results:")
            print(vif_results.to_string(index=False))
            
            high_vif = vif_results[vif_results['High_Multicollinearity']]
            if not high_vif.empty:
                print(f"\n⚠️  WARNING: {len(high_vif)} variables have VIF > {vif_threshold}")
                print("These variables may cause multicollinearity issues in modeling:")
                print(high_vif[['Variable', 'VIF']].to_string(index=False))
        else:
            print("VIF analysis could not be completed.")
        
        results['vif_analysis'] = vif_results
        
        # 4. Categorical Associations
        if self.categorical_cols:
            print(f"\n{'-'*50}")
            print("4. CATEGORICAL ASSOCIATIONS (Cramér's V)")
            print(f"{'-'*50}")
            
            cat_assoc = self.categorical_associations_analysis()
            if not cat_assoc.empty:
                print(f"\nTop categorical associations:")
                print(cat_assoc.head(10).to_string())
                
                strong_assoc = cat_assoc[cat_assoc > 0.5]
                if len(strong_assoc) > 0:
                    print(f"\nStrong associations (Cramér's V > 0.5):")
                    print(strong_assoc.to_string())
            else:
                print("No categorical associations calculated.")
            
            results['categorical_associations'] = cat_assoc
        
        # 5. Target Variable Analysis
        if target_var:
            print(f"\n{'-'*50}")
            print(f"5. TARGET VARIABLE ANALYSIS: {target_var}")
            print(f"{'-'*50}")
            
            try:
                # Feature importance
                importance = self.feature_importance_analysis(target_var)
                if not importance.empty:
                    print(f"\nTop 15 features by importance with {target_var}:")
                    print(importance.head(15).to_string(index=False))
                
                results['feature_importance'] = importance
                
                # Mixed variable analysis for key categorical variables
                if self.categorical_cols:
                    print(f"\nCategorical vs {target_var} analysis (top 5 categorical variables):")
                    for cat_var in self.categorical_cols[:5]:
                        try:
                            mixed_results = self.mixed_variable_analysis(cat_var, [target_var])
                            if not mixed_results.empty:
                                row = mixed_results.iloc[0]
                                print(f"  {cat_var}: F={row['F_statistic']:.2f}, "
                                     f"p={row['p_value']:.4f}, η²={row['eta_squared']:.3f}")
                        except:
                            continue
                            
            except Exception as e:
                print(f"Error in target variable analysis: {e}")
        
        # 6. Summary and Recommendations
        print(f"\n{'-'*50}")
        print("6. SUMMARY AND RECOMMENDATIONS")
        print(f"{'-'*50}")
        
        recommendations = []
        
        if not high_corr.empty:
            recommendations.append(f"• {len(high_corr)} pairs show high correlation (≥{correlation_threshold})")
            recommendations.append("  Consider feature selection or dimensionality reduction")
        
        if not vif_results.empty:
            high_vif_count = len(vif_results[vif_results['High_Multicollinearity']])
            if high_vif_count > 0:
                recommendations.append(f"• {high_vif_count} variables have high VIF (>{vif_threshold})")
                recommendations.append("  Consider removing or combining highly collinear features")
        
        if target_var and 'feature_importance' in results:
            imp_df = results['feature_importance']
            if not imp_df.empty:
                top_features = len(imp_df[imp_df.iloc[:, 1] > imp_df.iloc[:, 1].median()])
                recommendations.append(f"• {top_features} features show above-median importance with target")
                recommendations.append("  Focus on these for initial modeling")
        
        if not recommendations:
            recommendations.append("• Data shows good correlation structure for modeling")
            recommendations.append("• No major multicollinearity concerns detected")
        
        print("\nKey Findings:")
        for rec in recommendations:
            print(rec)
        
        results['recommendations'] = recommendations

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")

        if output_path:
            self.save_report(results, output_path)

        return results
    
    @staticmethod
    def _interpret_eta_squared(eta_sq: float) -> str:
        """Interpret eta-squared effect size"""
        if pd.isna(eta_sq):
            return "Cannot determine"
        elif eta_sq < 0.01:
            return "Very small"
        elif eta_sq < 0.06:
            return "Small"
        elif eta_sq < 0.14:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def _interpret_vif(vif_value: float) -> str:
        """Interpret VIF values"""
        if pd.isna(vif_value):
            return "Cannot calculate"
        elif vif_value < 5:
            return "Low multicollinearity"
        elif vif_value < 10:
            return "Moderate multicollinearity"
        else:
            return "High multicollinearity"

    @staticmethod
    def save_report(results: Dict, path: Union[str, Path]) -> None:
        """Save analysis results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="list")
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj

        serializable = {k: _convert(v) for k, v in results.items()}
        with path.open("w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Report saved to {path}")
