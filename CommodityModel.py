
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from typing import List, Tuple

from sklearn.preprocessing import MinMaxScaler

from CommodityData import CommodityData
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


class CommodityModel:
    def __init__(
        self,
        dataset: Type[CommodityData],
        models: List[Tuple[str, object]]
    ):
        self.dataset = dataset.data
        if self.dataset.empty:
            raise ValueError("No data available for modeling")
        self.models = models

        # Ensure each model in the list has 'fit' and 'predict' methods
        for model_name, model in self.models:
            if not callable(getattr(model, 'fit', None)) or not callable(getattr(model, 'predict', None)):
                raise ValueError(f"The model {model_name} must have 'fit' and 'predict' methods.")

        self.train_models()
        #self.predict(future_days=10)

    def preprocess_data(self):
        # Preprocess the dataset (ensure it contains the required columns)
        if "Date" not in self.dataset or "Close" not in self.dataset:
            raise KeyError("Required columns 'Date' and 'Close' not found in data")

        # Feature engineering: Convert Date to numeric days since the first date
        df = self.dataset.copy()[['Date', 'Close']].rename(columns={'Close': 'Price'})

        df.index = pd.to_datetime(df['Date'])
        df = df.asfreq('D').ffill()
        df["Days"] = (df["Date"] - df["Date"].min()).dt.days
        df.sort_values("Date", inplace=True)
        # Basic transformations
        df['Returns'] = df['Price'].pct_change()
        df['Log_Price'] = np.log(df['Price'])

        # Volatility measures
        df['Volatility_7'] = df['Returns'].shift(1).rolling(7).std()
        df['Volatility_21'] = df['Returns'].shift(1).rolling(21).std()

        # Moving averages - calculate separately
        windows = [3, 7, 14, 21, 50]
        for w in windows:
            df[f'MA_{w}'] = df['Price'].shift(1).rolling(w).mean()

        # Moving average ratios - calculate separately
        for w in windows:
            df[f'MA_Ratio_{w}'] = df['Price'] / df[f'MA_{w}']

        # Momentum indicators
        df['Momentum_7'] = df['Price'].shift(1).pct_change(7)
        df['Momentum_14'] = df['Price'].shift(1).pct_change(14)

        # Date features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month

        # Remove outliers
        q1 = df['Price'].quantile(0.05)
        q3 = df['Price'].quantile(0.95)
        df = df[(df['Price'] > q1) & (df['Price'] < q3)]

        df.dropna(inplace=True)
        df = df[df.columns].reset_index(drop=True)

        return df

    def train_models(self):
        # Prepare the data
        df = self.preprocess_data()
        scaler = MinMaxScaler()

        ml_inputs_num = list(set(df.columns) - set(['Date']))
        ml_inputs = list(set(df.columns) - set(['Price', 'Date']))

        df[ml_inputs_num] = scaler.fit_transform(df[ml_inputs_num])

        X = df[ml_inputs]
        y = df["Price"]

        # Time-based split: 80% train, 20% test
        split_index = int(len(df) * 0.9)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        test_dates = df["Date"].iloc[split_index:]  # Preserve test dates

        # Train each model and evaluate
        for model_name, model in self.models:
            print(f"Training model: {model_name}")

            model.fit(X_train, y_train)
            pred_train =model.predict(X_train)
            self.plot_results(y_train, pred_train, df["Date"].iloc[:split_index], model_name)
            # Evaluate the model
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            print(f"{model_name} - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")
            self.plot_results(y_test, predictions, test_dates, model_name)

    def predict(self, future_days: int):
        # Predict future values using each model
        if future_days <= 0:
            raise ValueError("future_days must be a positive integer")

        last_day = self.dataset["Days"].max()
        future_dates = pd.DataFrame({"Days": [last_day + i for i in range(1, future_days + 1)]})

        # Collect predictions from each model
        predictions = {}
        for model_name, model in self.models:
            try:
                future_predictions = model.predict(future_dates)  # Convert to NumPy array
                predictions[model_name] = future_predictions
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = None  # Handle errors gracefully

        return pd.DataFrame(predictions, index=future_dates["Days"])

    def plot_results(self, y_test, y_pred, test_dates, model_name):
        """ Plots Actual vs Predicted Prices """
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Actual vs Predicted Scatter Plot
        ax[0].scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicted')
        ax[0].plot(y_test, y_test, color='red', linestyle='dashed', label='Perfect Fit')
        ax[0].set_title('Actual vs Predicted Prices :'+model_name)
        ax[0].set_xlabel('Actual Prices')
        ax[0].set_ylabel('Predicted Prices')
        ax[0].legend()

        # Plot 2: Date vs Prices (Actual and Predicted)
        ax[1].plot(test_dates, y_test, color='black', label='Actual Prices')
        ax[1].plot(test_dates, y_pred, color='blue', linestyle='dashed',
                   label='Predicted Prices')
        ax[1].set_title('Date vs Prices (Actual & Predicted) :'+ model_name)
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Price')
        ax[1].legend()

        plt.tight_layout()
        plt.show()



