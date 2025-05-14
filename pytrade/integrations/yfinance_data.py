import yfinance as yf
import pandas as pd
import numpy as np


class YFinanceClient:

    @staticmethod
    def fetch_data(symbols: list[str]) -> tuple[pd.Series, pd.DataFrame]:
        data = yf.download(symbols)

        relevant_cols = [("Close", symbol) for symbol in symbols]
        data = data[relevant_cols].copy()
        data.columns = data.columns.get_level_values(level=1)

        last_prices = data.iloc[-1]

        # Compute returns
        data = np.log(data).diff(1)
        data = data.dropna(how="any", axis=0)

        return last_prices[symbols], data[symbols] # Enforce same order as passed list