import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from PARAMS import commodities, start_date, end_date, LOG
import yfinance as yf

class CommodityData:
    def __init__(
            self,
            commodity:str,
            commodity_mapping:dict,
            start_date:str,
            end_date:str,
    ):
        if not isinstance(commodity, str) or not commodity:
            raise ValueError("Commodity must be a non-empty string")
        if not isinstance(commodity_mapping, dict) or commodity not in commodity_mapping:
            raise ValueError("Commodity mapping must be a dictionary containing the given commodity")
        if not isinstance(start_date, str) or not isinstance(end_date, str):
            raise ValueError("start_date and end_date must be strings in YYYY-MM-DD format")

        self.commodity = commodity
        self.commodity_ticker = commodity_mapping[self.commodity]
        self.start_date = start_date
        self.end_date = end_date

        self.data = self.get_data()
        #self._line_plot(self.data)
        #self.seasonal_dependence()


    def get_data(self,
                 ) -> pd.DataFrame:
        LOG(f"Fetching data for {self.commodity} ({self.commodity_ticker})...")

        try:
            df = yf.download(
                self.commodity_ticker,
                start=self.start_date,
                end=self.end_date,
                interval="1d")

            if df.empty:
                LOG(f"No data found for {self.commodity}")
                return pd.DataFrame()

            df.reset_index(inplace=True)
            df["Commodity"] = self.commodity  # Add a column to distinguish commodities
            df.to_csv(self.commodity + ".csv", index=False)

            df = pd.read_csv(self.commodity + ".csv")
            df = df.iloc[1:].reset_index(drop=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df['Close'] = df["Close"].astype(float)
            return df

        except Exception as e:
            LOG(f"Error fetching data:{e}")
            return pd.DataFrame()


    def _line_plot(self, data):
        LOG("Plots")

        data["Date"] = pd.to_datetime(data["Date"])
        data['Close'] = data["Close"].astype(float)
        plt.figure(figsize=(12, 5))
        plt.plot(data["Date"], data["Close"], label=self.commodity)
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title("WTI Crude Oil Prices Over Time")
        plt.legend()
        plt.grid()
        plt.show()
        LOG("Done with plots")

    def seasonal_dependence(self):
        df = self.data.copy()
        df.index = pd.to_datetime(df["Date"])
        decomposition = seasonal_decompose(df['Close'], model = 'multiplicative', period = 365)
        decomposition.plot()
        plt.show()