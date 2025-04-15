import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load and preprocess data
file_path = "CMO-Historical-Data-Monthly.xlsx"
df = pd.read_excel(file_path, sheet_name='Monthly Prices')

# Data cleaning
df = df.iloc[3:].reset_index(drop=True)
df.columns = df.iloc[0].str.strip()
df = df[2:].reset_index(drop=True)
df.rename(columns={np.nan: 'Period'}, inplace=True)

for col in df.columns:
    df.rename(columns={col: col.replace(',', '')}, inplace=True)

df.rename(columns={'Period': 'Date'}, inplace=True)
columns = ["Date", "Crude oil Brent", "Natural gas Europe", "Cocoa",
           "Aluminum", "Copper", "Lead", "Nickel", "Zinc", "Platinum",
           "Wheat US HRW"]
df_monthly = df[columns]

# Convert Period to datetime
df_monthly['Date'] = pd.to_datetime(
    df_monthly['Date'].str.replace('M', '-'),
    format='%Y-%m'
)

# Convert prices to float
for col in df_monthly.columns[1:]:
    df_monthly[col] = pd.to_numeric(df_monthly[col], errors='coerce')


# Function to evaluate models
def evaluate_model(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred)
    }

# Train-test split and model evaluation
results = {}
forecast_steps = 12

for col in df_monthly.columns[1:]:
    try:
        # Prepare time series data (keep all data for modeling)
        ts_data = df_monthly[['Date', col]].dropna()
        ts_data = ts_data.set_index('Date').asfreq('MS')

        # Train-test split (last 12 months for testing)
        train = ts_data.iloc[:-12]
        test = ts_data.iloc[-12:]

        # Exponential Smoothing
        es_model = ExponentialSmoothing(
            train[col],
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()
        es_pred = es_model.forecast(steps = forecast_steps)
        results[col] = evaluate_model(test, es_pred)
        es_metrics = evaluate_model(test[col], es_pred)

        # SARIMA
        sarima_model = sm.tsa.SARIMAX(
            train[col],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        ).fit(disp=False)
        sarima_pred = sarima_model.forecast(steps=forecast_steps)
        sarima_metrics = evaluate_model(test[col], sarima_pred)

        # Store results
        results[col] = {
            'Exponential_Smoothing': es_metrics,
            'SARIMA': sarima_metrics
        }

        # For plotting only - filter data to >= 2020
        plot_data = ts_data[ts_data.index >= '2020-01-01']
        plot_train = train[train.index >= '2020-01-01']

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(plot_data.index, plot_data[col], label='Historical Data', color='black')
        plt.plot(test.index, es_pred, label='Exp Smoothing Forecast', color='blue', linestyle='--')
        plt.plot(test.index, sarima_pred, label='SARIMA Forecast', color='red', linestyle='-.')
        plt.title(f'{col} Forecast Evaluation\n'
                  f'ES: R²={es_metrics["R2"]:.2f}, MAE={es_metrics["MAE"]:.2f}\n'
                  f'SARIMA: R²={sarima_metrics["R2"]:.2f}, MAE={sarima_metrics["MAE"]:.2f}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xlim([pd.Timestamp('2020-01-01'), pd.Timestamp('2025-12-31')])
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing {col}: {str(e)}")
        continue

# Print summary results
print("\nModel Evaluation Summary:")
for commodity, metrics in results.items():
    print(f"\n{commodity}:")
    print("  Exponential Smoothing:")
    print(f"    R²: {metrics['Exponential_Smoothing']['R2']:.3f}")
    print(f"    MAE: {metrics['Exponential_Smoothing']['MAE']:.3f}")
    print(f"    MSE: {metrics['Exponential_Smoothing']['MSE']:.3f}")
    print("  SARIMA:")
    print(f"    R²: {metrics['SARIMA']['R2']:.3f}")
    print(f"    MAE: {metrics['SARIMA']['MAE']:.3f}")
    print(f"    MSE: {metrics['SARIMA']['MSE']:.3f}")