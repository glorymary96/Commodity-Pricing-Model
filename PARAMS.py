import datetime as dt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import lightgbm

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Models import *

FILES_DIR = '/home/glory/Documents/Python/Projects/PricingModel/Data/'

# Define commodities and their Yahoo Finance tickers
commodities = {
    "WTI_Crude_Oil": "CL=F",
    "Brent_Crude_Oil": "BZ=F",
    "Henry_Hub_Natural_Gas": "NG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Wheat": "ZW=F"
}

# models = [
#     ('Linear Regression', LinearRegression()),
#     ('Random Forest', RandomForestRegressor(random_state=42)),
#     ('Gradient Boost', GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)),
#     ('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
#     #('KNN', KNeighborsRegressor(n_neighbors=5)),
#     ('Decision Tree', DecisionTreeRegressor(random_state=42)),
#     ('Bagging Regressor', BaggingRegressor(n_estimators=150, random_state=42))
# ]

models = [
     # ("Exponential Smoothing", ExponentialSmoothingModel()),
     # ("SARIMA", SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,12))),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('LightGBM', lightgbm.LGBMRegressor())

]


# Define time range
start_date = "2020-01-01"  # Adjust based on data availability
end_date = dt.datetime.today().strftime('%Y-%m-%d')

def LOG(log_msg):
    if not isinstance(log_msg, str):
        raise ValueError("Log message must be a string")
    else:
        print(dt.datetime.now().strftime('%H:%M:%S')+ ' - '+ log_msg)


