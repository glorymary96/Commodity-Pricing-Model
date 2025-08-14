import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ExponentialSmoothingModel:
    def __init__(self, seasonal_periods=12, trend="add", seasonal="add"):
        self.model = None
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.fitted_values = None

    def fit(self, X, y):
        """Fit the Exponential Smoothing model"""
        try:
            # For time series, we typically only need y (the target variable)
            # X may contain datetime information which we can use as the index
            if isinstance(X, pd.DataFrame) and "Date" in X.columns:
                y.index = pd.to_datetime(X["Date"])

            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            ).fit()
            self.fitted_values = self.model.fittedvalues
            return self
        except Exception as e:
            print(f"Error fitting Exponential Smoothing model: {str(e)}")
            raise

    def predict(self, X):
        """Predict future values"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")

        try:
            # For time series forecasting, X can be the number of steps to predict
            if isinstance(X, (int, np.integer)):
                steps = X
            elif isinstance(X, pd.DataFrame):
                # If X is a DataFrame with dates, calculate how many steps to predict
                if "Date" in X.columns:
                    last_date = pd.to_datetime(self.fitted_values.index[-1])
                    future_dates = pd.to_datetime(X["Date"])
                    steps = len(future_dates[future_dates > last_date])
                else:
                    steps = len(X)
            else:
                steps = len(X)

            return self.model.forecast(steps)
        except Exception as e:
            print(f"Error predicting with Exponential Smoothing: {str(e)}")
            raise


class SARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.model = None
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_values = None

    def fit(self, X, y):
        """Fit the SARIMA model"""
        try:
            # For time series, we typically only need y (the target variable)
            # X may contain datetime information which we can use as the index
            if isinstance(X, pd.DataFrame) and "Date" in X.columns:
                y.index = pd.to_datetime(X["Date"])

            self.model = SARIMAX(
                y, order=self.order, seasonal_order=self.seasonal_order
            ).fit(disp=False)
            self.fitted_values = self.model.fittedvalues
            return self
        except Exception as e:
            print(f"Error fitting SARIMA model: {str(e)}")
            raise

    def predict(self, X):
        """Predict future values"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet")

        try:
            # For time series forecasting, X can be the number of steps to predict
            if isinstance(X, (int, np.integer)):
                steps = X
            elif isinstance(X, pd.DataFrame):
                # If X is a DataFrame with dates, calculate how many steps to predict
                if "Date" in X.columns:
                    last_date = pd.to_datetime(self.fitted_values.index[-1])
                    future_dates = pd.to_datetime(X["Date"])
                    steps = len(future_dates[future_dates > last_date])
                else:
                    steps = len(X)
            else:
                steps = len(X)

            return self.model.get_forecast(steps=steps).predicted_mean
        except Exception as e:
            print(f"Error predicting with SARIMA: {str(e)}")
            raise
