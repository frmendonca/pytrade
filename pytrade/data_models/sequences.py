
import pandas as pd
import numpy as np
from typing import Union, List
from pytrade.data_models.portfolio import Portfolio
from pytrade.data_models.option import Option
from pytrade.data_models.stock import Stock


def hedge_future_price(
    sampling_stock: pd.DataFrame,
    option_list: List[Option]
) -> pd.DataFrame:
    
    df = sampling_stock.stock_data[["returns"]]
    df = df.rename(columns = {"returns": sampling_stock.ticker})
    
    for option in option_list:
        current_underlying = option.underlying.stock.ticker_price
        ticker = option.underlying.ticker
        strike = option.strike
        
        df[f"future_underlying_{ticker}"] = current_underlying*(1 + df[sampling_stock.ticker])
        days_remaining = max(option.days_to_expiry - option.underlying.stock.return_freq, 1e-5)
        vectorized_black_scholes = np.vectorize(option.compute_black_scholes_put_option_price)

        df[f"future_premium_{ticker}_strike_{strike}"] = 100*vectorized_black_scholes(df[f"future_underlying_{ticker}"], days_remaining)*option.contracts

    return df


def compute_hedge_returns(
    df: pd.DataFrame,
    option_list: List[Option]
):
    for option in option_list:
        current_underlying = option.underlying.stock.ticker_price
        ticker = option.underlying.ticker
        strike = option.strike
        if f"future_underlying" not in df:
            df[f"future_underlying"] = current_underlying*(1 + df[ticker])

        days_remaining = max(option.days_to_expiry - option.underlying.stock.return_freq, 1e-5)
        vectorized_black_scholes = np.vectorize(option.compute_black_scholes_put_option_price)

        df[f"future_premium_strike_{strike}"] = 100*vectorized_black_scholes(df[f"future_underlying"], days_remaining)*option.contracts
        df[f"hedge_return_strike_{strike}"] = (df[f"future_premium_strike_{strike}"] - option.premium)/option.premium

    return df


def compute_total_return(
    df: pd.DataFrame,
    portfolio: Portfolio,
    option_list: Option,
    monthly_deposit: float = 2000
) -> pd.DataFrame:
    
    hedge_cost = sum(option.premium*option.contracts*100 for option in option_list)
    remainder_to_invest = max(monthly_deposit - hedge_cost, 0)

    new_portfolio_mkt_val = portfolio.market_value + remainder_to_invest

    # Apply returns
    df["eop_portfolio_mkt_val"] = new_portfolio_mkt_val*(1 + df["unhedged_returns"]) # TODO this is an approximation
    df["strategy_mkt_val"] = (
        df["eop_portfolio_mkt_val"] +
        df[[f"future_premium_{option.underlying.ticker}_strike_{option.strike}" for option in option_list]].sum(axis=1)
    )
    
    df["strategy_return"] = df["strategy_mkt_val"]/(portfolio.market_value+monthly_deposit) - 1

    return df


class Sequences:
    """
    Generates the return distribution of hedged returns
    """

    def __init__(
        self,
        portfolio: Portfolio,
        option_list: List[Option]
    ):
        self.portfolio = portfolio
        self.option_list = option_list
    

    def fit(
        self,
        sim_params: dict
    ) -> None:
        """
        Generate the return distribution taking into consideration
        the portfolio and option passed
        """

        # Fetch sampling distribution ticker info
        self.sampling_stock = Stock(ticker=sim_params.get("sampling_ticker", "SPX"), quantity = 0, beta = 0.0)
        self.sampling_stock.load_stock_data(freq = sim_params.get("freq", 1))

        # Get hedge future values
        df_seq_returns = hedge_future_price(self.sampling_stock, self.option_list)
        df_seq_returns["unhedged_returns"] = self.sampling_stock.stock_data["returns"]*self.portfolio.beta


        # Compute total return
        strategy_returns = compute_total_return(
            df_seq_returns,
            self.portfolio,
            self.option_list,
            sim_params.get("monthly_deposit", 2000)
        )

        self.strategy_returns = strategy_returns["strategy_return"].values
        strategy_returns.to_csv("Fitted_sequences.csv", index=False)
        
