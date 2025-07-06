
import pandas as pd
import typing as t
from enum import Enum
from pytrade.integrations.yfinance_data import YFinanceClient


class FxFactor(Enum):
    GBP = 0.01
    EUR = 1.0
    USD = 1.0


def unpack_portfolio(config: dict) -> dict[str, t.Any]:
    """
    Unpacks the portfolio information from a portfolio configuration

    :param portfolio_config: dict A dictionary representing the portfolio,
        expected to have a "stocks" key containing a list of Stock objects.

    :return dict with ticker as key and information as value
    """
    raw = {}
    fx_rates = []
    yf_client = YFinanceClient()

    # Iterate through the portfolio config    
    for pos in config:
        raw[pos.ticker] = {
            "quantity": pos.quantity,
            "currency": pos.currency,
            "dividend": pos.dividend,
        }

        ticker_currency = pos.currency.value
        if ticker_currency != "EUR":
            fx = f"{ticker_currency}EUR=X"
            fx_rates.append(fx)

        fx_rates = list(set(fx_rates))

    historical_fx_rates = yf_client.fetch(fx_rates) # Fetch FX rates

    # Fetch historical price data
    ticker_data = yf_client.fetch(list(raw.keys()))
    for ticker, ticker_data in ticker_data.items():

        currency = raw[ticker]["currency"]
        fx_factor = FxFactor[currency].value
        
        if currency != "EUR":
            fx_rate = pd.Series(historical_fx_rates[f"{currency}EUR=X"].data.Close, name = "FX")
        else:
            fx_rate = pd.Series(1.0, index = ticker_data.data.index, name = "FX")
        
        historical_data = ticker_data.data.join(fx_rate).assign(
            Close = lambda _df: _df["Close"]*_df["FX"]*fx_factor,
            Returns = lambda _df: _df["Close"].pct_change(1)
        )

        raw[ticker]["dividend"] *= fx_rate.iloc[-1]
        raw[ticker]["price"] = ticker_data.price * fx_factor * fx_rate.iloc[-1]
        raw[ticker]["data"] = historical_data[["Close", "Returns"]].dropna()

    return raw
