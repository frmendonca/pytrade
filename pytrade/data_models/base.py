
import pandas as pd
from dataclasses import dataclass
from yfinance import Ticker


@dataclass(repr = True)
class Stock:

    """
    A definition of a stock.
    """
    ticker: str
    historical_depth: int | None
    returns_freq: int | None
    
    def get_stock_data(self) -> pd.DataFrame:
        self.stock_data = (
            Ticker(self.ticker)
            .history(self._convert_year_to_period_str(self.historical_depth))
            .pipe(self._transform_stock_data, self.returns_freq)
        )
 

    def _convert_year_to_period_str(self, year: int | None = None) -> str:
        if year is None:
            return "100y"
        else:
            return str(year) + "y"


    def _transform_stock_data(self, df: pd.DataFrame, returns_freq: int = 1) -> pd.DataFrame:
        return (
            df
            [["Close"]]
            .assign(
                returns = lambda _df: _df["Close"].pct_change(returns_freq)
            )
            .query("returns.notna()", engine = "python")
        )
