
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from pytrade.data_models.base import Stock

@dataclass(repr=True)
class Portfolio:
    """
    A definition of a portfolio.
    A portfolio is composed of one or more tickers.
    - The total market value of a portfolio is given by the sum of the product
    between the market value of each ticker and the quantities held in the
    portfolio
    - The beta of a ticker is a measure of sensitivity to the broader market.
    For instance, a beta of 1.5 means that on average, if the SP500 moves 1%,
    that stock moves by 1.5%. Of course it is not a deterministic relationship, 
    as it is more than possible to have moves in excess (short of) 1.5 or even in
    opposite directions.

    :param shares: dict[str, float] defines the portfolio {ticker: quantity}
    :param beta: Optional[dict[str, float]] defines the betas for each stock. If not passed it is computed
    """

    shares: dict[str, float]
    betas: Optional[dict[str, float]]
    
    def build_portfolio(self) -> None:
        self.portfolio = {
            k: Stock(ticker = k, historical_depth = 10, returns_freq = 30)
            for k in self.shares.keys()
        }

    def _get_stock_data(self):
        for stock in self.portfolio.values():
            stock.get_stock_data()

    def _current_market_value(self) -> None:
        self.market_value = sum([v.stock_data["Close"].values[-1] for v in self.portfolio.values()])




portfolio = Portfolio(
    shares = {"AAPL": 1},
    betas = None
)
