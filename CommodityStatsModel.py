from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from CommodityData import CommodityData
from typing import Type, List, Tuple
import pandas as pd

import matplotlib.pyplot as plt

class CommodityStatsModel:
    def __init__(
        self,
        commodity: str,
        forecast_steps: int,
        dataset: Type[CommodityData],
        # models: List[Tuple[str, object]]
    ):
        self.commodity = commodity
        self.forecast_steps = forecast_steps
        self.dataset = dataset.data
        if self.dataset.empty:
            raise ValueError("No data available for modeling")
        self.commodity_data = self.preprocess_data()
        # self.models = models
        self.execute()

    def preprocess_data(self):
        # Preprocess the dataset (ensure it contains the required columns)
        if "Date" not in self.dataset or "Close" not in self.dataset:
            raise KeyError("Required columns 'Date' and 'Close' not found in data")

        # Feature engineering: Convert Date to numeric days since the first date
        self.dataset["Date"] = pd.to_datetime(self.dataset["Date"])
        commodity_data = self.dataset.copy()[['Date', 'Close']].dropna()
        return commodity_data

    def execute(self):
        forecasts_exp_smoothing ={}
        forecasts_sarima = {}
        try:
            commodity_series = self.commodity_data.set_index('Date')

            # Train-test split (last 12 months for testing)
            train = commodity_series.iloc[:-self.forecast_steps]
            test = commodity_series.iloc[-self.forecast_steps:]

            print(train.head())
            # Exponential Smoothing
            model_exp_smoothing = ExponentialSmoothing(train['Close'], seasonal='add', seasonal_periods=7)
            fit_exp_smoothing = model_exp_smoothing.fit()
            forecast_exp_smoothing = fit_exp_smoothing.forecast(steps=self.forecast_steps)
            metrics = self.evaluate_model(test['Close'], forecast_exp_smoothing)
            forecasts_exp_smoothing[self.commodity] = forecast_exp_smoothing

            # SARIMA (Seasonal ARIMA)
            model_sarima = SARIMAX(train['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fit_sarima = model_sarima.fit(disp=False)
            forecast_sarima = fit_sarima.forecast(steps=self.forecast_steps)
            forecasts_sarima[self.commodity] = forecast_sarima

            self.plot_results(forecasts_exp_smoothing, forecasts_sarima)

        except Exception as e:
            print(f"Forecasting failed for {self.commodity}: {e}")

    # Function to evaluate models
    def evaluate_model(self, y_true, y_pred):
        return {
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred)
        }

    def plot_results(self, forecasts_exp_smoothing, forecasts_sarima):
        plt.figure(figsize=(12, 6))

        plt.plot(self.commodity_data['Date'], self.commodity_data['Close'], label ="Historical Data", color = "black")
        # Plot Exponential Smoothing forecast
        forecast_dates = pd.date_range(self.commodity_data['Date'].iloc[-1], periods=self.forecast_steps + 1, freq='D')[1:]
        plt.plot(forecast_dates, forecasts_exp_smoothing[self.commodity], color='blue', linestyle='--', marker='o',
                 label='Exp. Smoothing Forecast')

        # Plot SARIMA forecast
        plt.plot(forecast_dates, forecasts_sarima[self.commodity], color='red', linestyle='-', marker='x', label='SARIMA Forecast')

        # Add plot details
        plt.title(f'{self.commodity} - 12-Month Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xlim([pd.Timestamp('2025-01-01'), pd.Timestamp('2026-12-31')])
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



