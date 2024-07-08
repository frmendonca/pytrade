
from pytrade.data_models.stock import Stock

class Portfolio:
    """
    The wrapper class for the portfolio

    :ivar config:dict the dictionary containing
    the configuration of the portfolio
    """

    def __init__(self, config: dict):
        self.config = config

        self.portfolio = []
        for ticker, ticker_kwargs in self.config.items():
            ticker_kwargs.update(
                {
                    "ticker": ticker
                }
            )

            self.portfolio.append(
                Stock(**ticker_kwargs)
            )

    
    def fit(self, **kwargs):
        for stock in self.portfolio:
            stock.load_stock_data(
                freq = kwargs["kwargs"].get("freq", 1)
            )

        # Compute market value
        self.market_value = sum(
            [
                stock.ticker_price*stock.quantity
                for stock in self.portfolio
            ]
        )