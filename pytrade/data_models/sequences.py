
import pandas as pd
import numpy as np
from typing import Union
from pytrade.data_models.portfolio import Portfolio
from pytrade.data_models.option import Option

def merge_portfolio_data(
        portfolio: Portfolio,
        option: Union[Option, None] = None
) -> pd.DataFrame:
    merged_returns = pd.DataFrame()
    for stock in portfolio.portfolio:
        if merged_returns.empty:
            merged_returns = stock.stock_data[["returns"]].copy()
            merged_returns.rename(columns = {"returns": stock.ticker}, inplace = True)
        else:
            merged_returns = merged_returns.join(
                stock.stock_data[["returns"]].rename(columns = {"returns": stock.ticker}),
                how = "inner"
            )
    
    if option is not None:
        merged_returns = merged_returns.join(
            option.underlying.stock.stock_data[["returns"]].rename(columns = {"returns": option.underlying.ticker}),
            how = "inner"
        )

    return merged_returns.reset_index(drop = True)


def compute_unhedged_returns(
    df: pd.DataFrame,
    portfolio: Portfolio
) -> pd.DataFrame:
    
    '''
    Compute the total hedged return according to the option 
    passed

    :param df: pd.DataFrame containing the returns of assets in portfolio
    and if relevant, the returns of the underlying asset
    :param portfolio: Portfolio containing the relevant assets
    :returns pd.DataFrame with unhedged returns
    '''
        
    df = df.assign(
            unhedged_return = lambda _df: sum(
                _df[stock.ticker]*stock.market_value for stock in portfolio.portfolio
            )/portfolio.market_value
        )
    
    return df


def compute_hedge_returns(
    df: pd.DataFrame,
    option: Option
):
    current_underlying = option.underlying.stock.ticker_price
    df["future_underlying"] = current_underlying*(1 + df[option.underlying.ticker])

    days_remaining = max(option.days_to_expiry - option.underlying.stock.return_freq, 1e-5)
    vectorized_black_scholes = np.vectorize(option.compute_black_scholes_put_option_price)

    df["future_premium"] = vectorized_black_scholes(df["future_underlying"], days_remaining)
    df["hedge_return"] = (df["future_premium"] - option.premium)/option.premium
    return df


def compute_total_return(
    df: pd.DataFrame,
    portfolio: Portfolio,
    option: Option,
    option_contracts: int,
    monthly_deposit: float = 2000
) -> pd.DataFrame:
    
    hedge_cost = option.premium*option_contracts*100
    remainder_to_invest = max(monthly_deposit - hedge_cost, 0)

    new_portfolio_mkt_val = portfolio.market_value + remainder_to_invest

    # Apply returns
    df["eop_portfolio_mkt_val"] = new_portfolio_mkt_val*(1 + df["unhedged_return"]) # TODO this is an approximation
    df["strategy_mkt_val"] = df["eop_portfolio_mkt_val"] + df["future_premium"]*option_contracts*100
    df["strategy_return"] = (df["strategy_mkt_val"] - 100*option_contracts*option.premium)/(portfolio.market_value+monthly_deposit) - 1

    return df


class Sequences:
    """
    Generates the return distribution of hedged returns
    """

    def __init__(
        self,
        portfolio: Portfolio,
        option: Union[Option, None] = None
    ):
        self.portfolio = portfolio
        self.option = option
    

    def fit(
        self,
        option_contracts: int = 0,
        **kwargs: dict
    ) -> None:
        """
        Generate the return distribution taking into consideration
        the portfolio and option passed
        """

        # Merge portfolio stock data with options underlying stock data
        merged_returns = merge_portfolio_data(self.portfolio, self.option)
        merged_returns = merged_returns.pipe((compute_unhedged_returns, "df"), portfolio=self.portfolio)
        
        # Get hedge returns
        hedge_returns = compute_hedge_returns(merged_returns, self.option)

        # Compute total return
        strategy_returns = compute_total_return(
            hedge_returns,
            self.portfolio,
            self.option,
            option_contracts,
            kwargs["kwargs"].get("monthly_deposit", 2000)
        )

        self.strategy_returns = strategy_returns["strategy_return"].values