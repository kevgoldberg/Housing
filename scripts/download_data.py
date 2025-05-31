# This script downloads the Kaggle Housing data into data/raw/
import os
import kaggle
import zipfile

data_dir = os.path.join(os.path.dirname(__file__), '../data/raw')
os.makedirs(data_dir, exist_ok=True)

# Download the House Prices - Advanced Regression Techniques dataset
kaggle.api.authenticate()
kaggle.api.competition_download_files('house-prices-advanced-regression-techniques', path=data_dir)

# Unzip the downloaded file
zip_path = os.path.join(data_dir, 'house-prices-advanced-regression-techniques.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

print('Download and extraction complete. Files are in data/raw/')
