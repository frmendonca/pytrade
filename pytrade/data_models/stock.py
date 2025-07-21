import pandas as pd
import yfinance as yf
from pytrade.data_models.base import Currency
from pytrade.utils.utils import compute_stock_returns

class Stock:
    """
    The wrapper class for a stock.

    :ivar ticker: str the symbol of the stock
    :ivar quantity: int the number of shares for a given stock
    :ivar cost_basis the average cost of the stock lot
    :ivar dividend the annual dividend paid by the company
    :ivar currency the currency of the stock
    """

    def __init__(
        self,
        ticker: str,
        quantity: int,
        cost_basis: float = 0.0,
        dividend: float = 0.0,
        currency: Currency = Currency.USD
    ) -> None:
        
        self.ticker = ticker
        self.quantity = quantity
        self.cost_basis = cost_basis
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


