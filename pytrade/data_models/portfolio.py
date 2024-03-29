
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from pytrade.data_models.stock import Stock

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
    betas: Optional[dict[str, float]] = None
    
    def fit_portfolio(self) -> None:
        self.portfolio = {
            ticker: Stock(ticker = ticker)
            for ticker in self.shares.keys()
        }

        self._get_stock_data()
        self._compute_market_value()
        self._compute_portfolio_betas()

    def _get_stock_data(self):
        for stock in self.portfolio.values():
            stock.get_stock_data()

    def _compute_market_value(self) -> None:
        self.market_value = sum(
            [
                self.portfolio[ticker].stock_data["Close"].values[-1].item()*self.shares[ticker]
                for ticker in self.shares.keys()
            ]
        )

    def _compute_portfolio_betas(self) -> None:
        if self.betas == None:
            self.portfolio_beta = 1
        else:
            self.portfolio_beta = sum(
                [
                    self.portfolio[ticker].stock_data["Close"].values[-1].item()*self.shares[ticker]*self.betas[ticker]
                    for ticker in self.shares.keys()
                ]
            )/self.market_value
        
