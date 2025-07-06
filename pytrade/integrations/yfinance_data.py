import yfinance as yf
from typing import Generator, Any
from pytrade.integrations.base import Ticker


DATA_PARAMS = {
    "returns": {
        "frequencies": [1, 7, 30, 365]
    },
    "indicators": {
        "sma": {
            "frequencies": [7, 50, 200]
        }
    }
}


class YFinanceClient:
    
    def fetch(self, symbols: list[str]) -> dict[str,Ticker]:
        ticker_objects = {}
        for ticker_obj in self._fetch_data(symbols):
            if ticker_obj.symbol != 'UNKNOWN':
                ticker_objects[ticker_obj.symbol] = ticker_obj

        return ticker_objects


    def _fetch_data(self, symbols: list[str]) -> Generator[Ticker, Any, Any]:
        data = yf.download(symbols)

        for sym in symbols:
            sym_df = data[[("Close", sym)]].dropna()
            sym_df.columns = ["Close"]

            sym_df = (
                sym_df
                .assign(
                    **{f"returns_freq_{d}_days": sym_df["Close"].pct_change(d) for d in DATA_PARAMS["returns"]["frequencies"]}
                )
                .assign(
                    **{f"sma_freq_{d}_days": sym_df["Close"].rolling(window = d).mean() for d in DATA_PARAMS["indicators"]["sma"]["frequencies"]}
                )
            )

            yield Ticker(
                symbol = sym,
                price = sym_df["Close"].iloc[-1],
                data = sym_df
            )