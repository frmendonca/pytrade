import yfinance as yf
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed


class YFinanceClient:
    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols

    @staticmethod
    def _fetch_data(symbol) -> tuple[str, pd.DataFrame]:
        data = yf.download(symbol)
        data.columns = data.columns.get_level_values(level=0)
        data = data.assign(returns=lambda _df: np.log(_df["Close"]).diff(1)).dropna(
            how="any", axis=0
        )

        return symbol, data

    def fetch_data(self) -> dict[str, pd.DataFrame]:
        results = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._fetch_data, symbol) for symbol in self.symbols
            }
            for future in as_completed(futures):
                symbol, data = future.result()
                results[symbol] = data

        return results
