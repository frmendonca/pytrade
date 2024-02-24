
import pandas as pd
from dataclasses import dataclass
from yfinance import Ticker


@dataclass(repr = True)
class Stock:

    """
    A definition of a stock.
    """
    ticker: str
    historical_depth: str = "1y"
    returns_freq: int = 1
    
    def get_stock_data(self) -> pd.DataFrame:
        self.stock_data = (
            Ticker(self.ticker)
            .history(self.historical_depth)
            .pipe(self._compute_returns, self.returns_freq)
        )

    def _compute_returns(self, df: pd.DataFrame, returns_freq: int = 1) -> pd.DataFrame:
        return (
            df
            [["Close"]]
            .assign(
                returns = lambda _df: _df["Close"].pct_change(returns_freq)
            )
            .query("returns.notna()", engine = "python")
        )