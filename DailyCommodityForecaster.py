from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Type
from CommodityData import CommodityData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RobustDailyCommodityForecaster:
    def __init__(self, commodity: str, forecast_days: int, dataset: pd.DataFrame):
        self.commodity = commodity
        self.forecast_days = forecast_days
        self.dataset = (Type[CommodityData],)
        self.results = None

        self.dataset = dataset.data
        # Validate data
        if self.dataset.empty:
            raise ValueError("No data available for modeling")
        if "Date" not in self.dataset.columns or "Close" not in self.dataset.columns:
            raise KeyError("Dataset must contain 'Date' and 'Close' columns")

        self._preprocess_data()
        self._execute_forecasting()

    def _preprocess_data(self):
        """Prepare and validate time series data"""
        self.dataset["Date"] = pd.to_datetime(self.dataset["Date"])
        self.dataset = (
            self.dataset[["Date", "Close"]]
            .dropna()
            .set_index("Date")
            .asfreq("D")
            .ffill()
        )

    def _execute_forecasting(self):
        """Run forecasting pipeline with convergence handling"""
        # Train-test split (last 30 days for testing)
        train = self.dataset.iloc[:-30]
        test = self.dataset.iloc[-30:]

        # Initialize results storage
        forecasts = {
            "dates": pd.date_range(
                start=train.index[-1] + pd.Timedelta(days=1),
                periods=self.forecast_days,
                freq="D",
            ),
            "actual": test["Close"],
            "train": train["Close"],
        }

        # 1. Exponential Smoothing with convergence handling
        try:
            es_model = ExponentialSmoothing(
                train["Close"],
                trend="add",
                seasonal="add",
                seasonal_periods=7,
                initialization_method="heuristic",  # More stable initialization
                freq="D",
            ).fit(
                optimized=True,
                use_brute=True,  # Brute-force optimization for better convergence
                remove_bias=True,  # Reduce systematic forecast errors
            )
            forecasts["es"] = es_model.forecast(self.forecast_days)
            forecasts["es_metrics"] = self._calculate_metrics(
                test["Close"], forecasts["es"][: len(test)]
            )
        except Exception as e:
            print(f"Exponential Smoothing failed: {str(e)}")
            forecasts["es"] = None
            forecasts["es_metrics"] = None

        # 2. SARIMA with error handling
        try:
            sarima_model = SARIMAX(
                train["Close"],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(
                disp=0,
                maxiter=1000,  # Increased iterations
                method="nm",  # Nelder-Mead optimization
            )
            sarima_forecast = sarima_model.get_forecast(steps=self.forecast_days)
            forecasts["sarima"] = sarima_forecast.predicted_mean
            forecasts["sarima_ci"] = sarima_forecast.conf_int()
            forecasts["sarima_metrics"] = self._calculate_metrics(
                test["Close"], forecasts["sarima"][: len(test)]
            )
        except Exception as e:
            print(f"SARIMA failed: {str(e)}")
            forecasts["sarima"] = None
            forecasts["sarima_metrics"] = None

        self.results = forecasts
        self._plot_results()

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics with alignment check"""
        y_true = y_true[: len(y_pred)]  # Ensure equal length
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "Direction_Accuracy": (
                np.sign(y_pred.diff().fillna(1)) == np.sign(y_true.diff().fillna(1))
            ).mean(),
        }

    def _plot_results(self):
        """Generate visualization of results"""

        print("Plotting results")
        if self.results is None:
            print("No results to plot")
            return

        plt.figure(figsize=(14, 7))

        # Plot training data (last 90 days)
        plt.plot(
            self.results["train"].index[-90:],
            self.results["train"][-90:],
            label="Training Data",
            color="blue",
        )

        # Plot test data
        plt.plot(
            self.results["actual"].index,
            self.results["actual"],
            label="Actual Values",
            color="green",
        )

        # Plot Exponential Smoothing forecast if available
        if self.results["es"] is not None:
            plt.plot(
                self.results["dates"],
                self.results["es"],
                label=f'Exp Smoothing (R²={self.results["es_metrics"]["R2"]:.2f})',
                color="red",
                linestyle="--",
            )

        # Plot SARIMA forecast if available
        if self.results["sarima"] is not None:
            plt.plot(
                self.results["dates"],
                self.results["sarima"],
                label=f'SARIMA (R²={self.results["sarima_metrics"]["R2"]:.2f})',
                color="purple",
                linestyle="-.",
            )
            # Add confidence interval
            plt.fill_between(
                self.results["dates"],
                self.results["sarima_ci"].iloc[:, 0],
                self.results["sarima_ci"].iloc[:, 1],
                color="purple",
                alpha=0.1,
                label="SARIMA 95% CI",
            )

        plt.title(f"{self.commodity} - {self.forecast_days}-Day Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage:
# Assuming you have a DataFrame with 'Date' and 'Close' columns
# forecaster = RobustDailyCommodityForecaster(
#     commodity="Gold",
#     forecast_days=14,
#     dataset=your_dataframe
# )
