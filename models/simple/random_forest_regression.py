"""Random Forest Regression Model for House Price Prediction"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Path to cleaned data (update if needed)
DATA_PATH = os.path.join('..', '..', 'data', 'processed', 'V2', 'train_advanced_cleaned.csv')
MODEL_SAVE_PATH = os.path.join('..', 'simple', 'random_forest_model.pkl')


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    if 'SalePrice' not in df.columns:
        raise ValueError('SalePrice column not found in data.')
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f'Random Forest RMSE: {rmse:.2f}')

    # 5. Save model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')


if __name__ == '__main__':
    main()
