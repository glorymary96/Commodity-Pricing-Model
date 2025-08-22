# Commodity Pricing Forecasting Framework

### Forecasting commodity prices using statistical models and machine learning

---

##  Overview

This project provides a structured framework for training, evaluating, and forecasting multiple commodity prices using configurable statistical and ML models.

**Supported Commodities**: Brent Crude Oil, WTI Crude Oil, Gold, Natural Gas, Wheat, and more.

---

##  Key Highlights

- 🛠 **Modular Architecture** — Separate components for data ingestion, modeling, statistical analysis, and forecasting.
- ⚙ **Configurable via `PARAMS.py`** — Easily switch commodities, date ranges, model settings, and feature options.
-  **Visual and Quantitative Outputs** — Evaluation metrics (MAE, MSE, R²) and actual vs forecast visualizations.
-  **Exploratory Notebook (`data_check.ipynb`)** — Quickly assess data shape and quality before modeling.

---

##  Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/glorymary96/Commodity-Pricing-Model.git
cd Commodity-Pricing-Model
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure Parameters

Open and edit PARAMS.py to set:

    - Commodities to analyze (commodities = {...})
    - Date range (start_date, end_date)

4. Run Forecast WorkFlow
```bash
python main.py
```
This will:
- Load fetch historical price data from yahoo finance and create a csv file
- Train models per commodity and time frequency
- Generate evaluation metrics and visual output (Predicted, Actual)
-  Future price forecasts

5. Explore & Evaluate

Visuals and metrics are displayed.

6. Project Architecture
```
├── CommodityData.py           # Data ingestion & preprocessing
├── CommodityModel.py          # Model training logic
├── CommodityStatsModel.py     # Statistical modeling functions
├── DailyCommodityForecaster.py# Daily forecasting pipeline
├── StatsModel_Monthly.py      # Monthly stats & forecasting
├── Models.py                  # Shared model definitions
├── data_check.ipynb           # Exploratory data analysis
├── main.py                    # Central orchestration script
├── PARAMS.py                  # Configuration & settings
├── requirements.txt           # Python dependencies
├── *.csv                      # Historical price data files
```

7. Customization Options

You can extend the framework by:
- Adding more commodities in PARAMS.py or uploading corresponding CSVs
- Adjust date ranges
- Customizing model types
- Add custom feature engineering
- Incorporating more evaluation metrics
- Adding support for multi-step forecasting or real-time updates

