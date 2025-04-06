#
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
#
# # Define the file path
# file_path = "CMO-Historical-Data-Monthly.xlsx"
#
# # Read the Excel file
# df =pd.read_excel(file_path, sheet_name='Monthly Prices')
# df = df.iloc[3:].reset_index(drop=True)
# df.columns = df.iloc[0].str.strip()
# df = df[2:].reset_index(drop=True)  # Remove the first row and reset index
#
# df.rename(columns={np.nan: 'Period'}, inplace=True)
#
# for col in df.columns:
#     df.rename(columns={col:col.replace(',', '')}, inplace=True)
#
# columns = ["Period",
#     "Crude oil Brent", "Natural gas Europe", "Cocoa",
#     "Aluminum", "Copper", "Lead", "Nickel", "Zinc", "Platinum"
# ]
#
# df_monthly = df[columns]
# df_monthly['Period'] = pd.to_datetime(
#     df_monthly['Period'].str.replace('M', '-'),
#     format='%Y-%m'
# )
# df_monthly.rename(columns={'Period': 'Date'}, inplace=True)
#
# for col in df_monthly.columns[1:]:
#     df_monthly[col] = df_monthly[col].astype(float)
#
# # Step 5: Set up forecast parameters and dictionaries to store results
# forecast_steps = 12
# forecasts_exp_smoothing = {}
# forecasts_sarima = {}
#
# # Step 6: Forecasting with Exponential Smoothing and SARIMA for each commodity
# for col in df_monthly.columns[1:]:
#     try:
#         # Prepare the time series data
#         commodity_df = df_monthly[['Date', col]].dropna()
#         commodity_series = commodity_df.set_index('Date')[col]
#
#         # Exponential Smoothing
#         model_exp_smoothing = ExponentialSmoothing(commodity_series, seasonal='add', seasonal_periods=12)
#         fit_exp_smoothing = model_exp_smoothing.fit()
#         forecast_exp_smoothing = fit_exp_smoothing.forecast(steps=forecast_steps)
#         forecasts_exp_smoothing[col] = forecast_exp_smoothing
#
#         # SARIMA (Seasonal ARIMA)
#         model_sarima = sm.tsa.SARIMAX(commodity_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
#         fit_sarima = model_sarima.fit(disp=False)
#         forecast_sarima = fit_sarima.forecast(steps=forecast_steps)
#         forecasts_sarima[col] = forecast_sarima
#
#     except Exception as e:
#         print(f"Forecasting failed for {col}: {e}")
#
# # Step 7: Plot the forecasts
# for col in df_monthly.columns[1:]:
#     if col in forecasts_exp_smoothing and col in forecasts_sarima:
#         plt.figure(figsize=(12, 6))
#
#         # Plot historical data
#         commodity_df = df_monthly[['Date', col]].dropna()
#         # Assuming commodity_df has a 'Date' column in datetime format
#         plt.plot(commodity_df['Date'], commodity_df[col], label='Historical Data', color='black', linewidth=2)
#
#         # Plot Exponential Smoothing forecast
#         forecast_dates = pd.date_range(commodity_df['Date'].iloc[-1], periods=forecast_steps + 1, freq='M')[1:]
#         plt.plot(forecast_dates, forecasts_exp_smoothing[col], color='blue', linestyle='--', marker='o',
#                  label='Exp. Smoothing Forecast')
#
#         # Plot SARIMA forecast
#         plt.plot(forecast_dates, forecasts_sarima[col], color='red', linestyle='-', marker='x', label='SARIMA Forecast')
#
#         # Add plot details
#         plt.title(f'{col} - 12-Month Price Forecast')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.xlim([pd.Timestamp('2020-01-01'), pd.Timestamp('2026-12-31')])
#         plt.legend(loc='upper left')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()