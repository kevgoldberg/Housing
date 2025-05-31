# --------------------------------------------------------------------------------
# # Data Cleaning Notebook
# This notebook covers the data cleaning process for the housing dataset. Steps include loading the data, inspecting for missing values, handling outliers, and preparing the data for modeling.

# Import required libraries
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------
# # Load the training data
# Load the training data from the raw data folder.

# Load the training data
train_df = pd.read_csv('../../data/raw/train.csv')
train_df.head()

# --------------------------------------------------------------------------------
# ## Inspect Missing Values
# Check for missing values in the dataset to identify columns that need cleaning or imputation.

# Display missing value counts
missing = train_df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

# --------------------------------------------------------------------------------
# ## Handle Missing Values
# Decide on strategies for handling missing values (e.g., imputation, removal) based on the data and domain knowledge.

# Fill missing numerical values with median
num_cols = train_df.select_dtypes(include=[np.number]).columns
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())

# Fill missing categorical values with mode
cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])

# Convert categorical columns to numerical using label encoding
for col in cat_cols:
    train_df[col] = train_df[col].astype('category').cat.codes

# --------------------------------------------------------------------------------
# ## Detect and Handle Outliers
# Visualize and handle outliers as needed.

# Visualize outliers in 'SalePrice'
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
train_df['SalePrice'].plot(kind='box')
plt.title('SalePrice Outliers')
plt.show()

# --------------------------------------------------------------------------------
# ## Save Cleaned Data
# Export the cleaned dataset for further analysis or modeling.

# Save cleaned data
import os
os.makedirs('../../data/processed/V1', exist_ok=True)
train_df.to_csv('../../data/processed/V1/train_cleaned.csv', index=False)



