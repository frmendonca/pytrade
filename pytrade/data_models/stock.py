
import pandas as pd
import yfinance as yf


class Stock:
    """
    The wrapper class for a stock.
    
    :ivar ticker: str the symbol of the stock
    :ivar quantity: int the number of shares for a given stock
    :ivar beta: float the beta for the stock
    """

    def __init__(self, ticker: str, quantity: int, beta: float) -> None:
        self.ticker=ticker
        self.quantity=quantity
        self.beta=beta

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(ticker={self.ticker}, quantity={self.quantity}, beta={self.beta})"


    def load_stock_data(self, freq: int  = 1):
        '''
        Loads the stock data for each ticker in portfolio
        '''
        self.stock_data = (
            yf.Ticker(self.ticker)
            .history("100y")
            .pipe(compute_stock_returns, freq=freq)
        )
        self.stock_data.index = self.stock_data.index.date
        
        self.ticker_price = self.stock_data["Close"].values[-1].item()
        self.market_value = self.ticker_price*self.quantity
        self.return_freq = freq


def compute_stock_returns(df: pd.DataFrame, freq: int = 1): 
    return (
        df
        [["Close"]]
        .assign(returns = df["Close"].pct_change(freq))
        .query("returns.notna()")
    )

