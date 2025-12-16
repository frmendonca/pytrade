import yfinance as yf
from typing import Generator, Any
from pytrade.integrations.base import Ticker


class YFinanceClient:
    
    def fetch(self, symbols: list[str], **kwargs) -> dict[str,Ticker]:
        ticker_objects = {}
        for ticker_obj in self._fetch_data(symbols, **kwargs):
            if ticker_obj.symbol != 'UNKNOWN':
                ticker_objects[ticker_obj.symbol] = ticker_obj

        return ticker_objects


    def _fetch_data(self, symbols: list[str], **kwargs) -> Generator[Ticker, Any, Any]:
        data = yf.download(symbols, **kwargs)

        for sym in symbols:
            sym_df = data[[("Close", sym), ("Open", sym), ("High", sym), ("Low", sym)]].dropna()
            sym_df.columns = ["Close", "Open", "High", "Low"]

            yield Ticker(
                symbol = sym,
                price = sym_df["Close"].iloc[-1],
                data = sym_df
            )