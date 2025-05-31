# Import required libraries
import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv('../../data/raw/train.csv')
train_df.head()

# Display missing value counts
missing = train_df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

# Fill structural missing values with 'None' for relevant categorical features
structural_none_features = ['Alley', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                            'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
for col in structural_none_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna('None')

# Predictive imputation for a categorical column (e.g., GarageType)
feature = 'GarageType'
if train_df[feature].isnull().sum() > 0:
    not_null = train_df[train_df[feature].notnull()]
    null = train_df[train_df[feature].isnull()]
    predictors = [col for col in train_df.columns if col not in [feature, 'SalePrice']]
    X = pd.get_dummies(not_null[predictors], dummy_na=True)
    X_null = pd.get_dummies(null[predictors], dummy_na=True)
    X_null = X_null.reindex(columns=X.columns, fill_value=0)
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X, not_null[feature])
    train_df.loc[train_df[feature].isnull(), feature] = clf.predict(X_null)

# Define quality-related features and their order
quality_map = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ordinal_features = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
]
# Ensure all ordinal features are present in the data
ordinal_features = [col for col in ordinal_features if col in train_df.columns]
print(f"Ordinal features found: {ordinal_features}")

# Identify numeric and categorical columns (excluding target and ordinal features)
numeric_features = train_df.select_dtypes(include=[np.number]).columns.drop('SalePrice', errors='ignore')
categorical_features = [col for col in train_df.select_dtypes(include='object').columns 
                       if col not in ordinal_features]
print(f"Numeric features: {len(numeric_features)}")
print(f"Ordinal features: {len(ordinal_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Define preprocessing pipelines for each feature type
numeric_pipeline = Pipeline([
    ('imputer', IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50),
        max_iter=10, random_state=0)),
    ('scaler', StandardScaler())
])
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('ordinal', OrdinalEncoder(categories=[quality_map]*len(ordinal_features), handle_unknown='use_encoded_value', unknown_value=-1))
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
# Combine all pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('ord', ordinal_pipeline, ordinal_features),
    ('cat', categorical_pipeline, categorical_features)
])
# Fit and transform the data (excluding target)
X_clean = preprocessor.fit_transform(train_df.drop(columns='SalePrice'))
print(f"Preprocessed data shape: {X_clean.shape}")

# Convert preprocessed data to DataFrame with feature names
# Get feature names for each transformer
num_cols = list(numeric_features)
ord_cols = list(ordinal_features)
cat_cols = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
all_cols = num_cols + ord_cols + list(cat_cols)

import scipy.sparse
if scipy.sparse.issparse(X_clean):
    X_clean_dense = X_clean.A
else:
    X_clean_dense = X_clean

X_clean_df = pd.DataFrame(X_clean_dense, columns=all_cols, index=train_df.index)
print(X_clean_df.head())
print(f"All features are now numeric. Shape: {X_clean_df.shape}")

# Save cleaned numeric data
os.makedirs('../../data/processed/V2', exist_ok=True)
X_clean_df.to_csv('../../data/processed/V2/train_advanced_cleaned.csv', index=False)
print(f"Cleaned numeric dataset saved with shape: {X_clean_df.shape}")

