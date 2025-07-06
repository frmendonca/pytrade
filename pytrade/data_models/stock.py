import pandas as pd
import yfinance as yf
import typing as t
from pytrade.data_models.base import Currency

class Stock:
    """
    The wrapper class for a stock.

    :ivar ticker: str the symbol of the stock
    :ivar quantity: int the number of shares for a given stock
    :ivar beta: float the beta for the stock
    """

    def __init__(
        self,
        ticker: str,
        quantity: int,
        avg_cost: float = 0.0,
        dividend: float = 0.0,
        currency: Currency = Currency.USD
    ) -> t.Self:
        
        self.ticker = ticker
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.dividend = dividend
        self.currency = currency

        self.stock_data = None
        self.ticker_price = None
        self.return_freq = None


    def __repr__(self):
        cls = self.__class__.__name__
        return (
            f"{cls}(ticker={self.ticker}, quantity={self.quantity}, currency={self.currency.value})"
        )

    def load_stock_data(self, freq: int = 1):
        """
        Loads the stock data for each ticker in portfolio
        """
        self.stock_data = (
            yf.Ticker(self.ticker)
            .history("100y")
            .pipe(compute_stock_returns, freq=freq)
        )
        self.stock_data.index = self.stock_data.index.date

        self.ticker_price = self.stock_data["Close"].values[-1].item()
        self.market_value = self.ticker_price * self.quantity
        self.return_freq = freq


def compute_stock_returns(df: pd.DataFrame, freq: int = 1):
    return (
        df[["Close"]]
        .assign(returns=df["Close"].pct_change(freq))
        .query("returns.notna()")
    )
