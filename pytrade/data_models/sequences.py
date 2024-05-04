
import numpy as np

from pytrade.data_models.options import Option
from pytrade.data_models.stock import Stock
from pytrade.data_models.portfolio import Portfolio
from pytrade.utils import (
    beta_weight_return_distribution,
    compute_future_underlying,
    compute_hedge_allocation,
    compute_hedge_return,
    simulate_iv_change
    )

class Sequences():

    """
    Generates random draws to simulate a distribution of returns
    """

    def __init__(
            self,
            base_ticker: str = "SPY",
            base_historical_depth: str = "25y",
            base_returns_freq: int = 30,
            portfolio: Portfolio = None,
            option: Option = None
    ) -> None:
        
        """
        :param base_ticker: the string ticker name of the base asset used to draw returns
        :param base_historical_depth: how far back should we get data for the ticker
        :param return_freq: frequence of returns in sequence
        :param option: an Option object to mix with base_ticker return distribution
        """
        self._base_ticker = base_ticker
        self._base_historical_depth = base_historical_depth
        self._base_returns_freq = base_returns_freq
        self._portfolio = portfolio
        self._option = option
        self._specs = None


    def fit(self) -> None:
        """
        This method fits the sequences
        """

        # As a first step, load and fetch the data for the base ticker used in the simulation
        # This loads an object Stock that identifies the base ticker historical data
        # We can access this data with self._specs.stock_data
        self._get_initial_specs()

        return_sequence = beta_weight_return_distribution(
            self._specs.stock_data["returns"], self._portfolio.portfolio_beta
        )

        if (self._option.contracts == 0) | (self._option == None):
            self.sequence = return_sequence
        else:
            if self._portfolio == None:
                raise RuntimeError("Requires a portfolio to compute sequences")
            
            hedge_allocation = compute_hedge_allocation(
                self._portfolio.market_value,
                self._option.contracts,
                self._option.premium
            )

            underlying_current = self._specs.stock_data["Close"].values[-1]
            underlying_future = compute_future_underlying(underlying_current, return_sequence)
            self._option.days_expiry -= self._base_returns_freq

            # Simulate new option IV
            new_options = simulate_iv_change(return_sequence, self._option)
            hedge_return = compute_hedge_return(new_options, underlying_future)

            self.sequence = (1 - hedge_allocation)*return_sequence + hedge_allocation*hedge_return


    def _get_initial_specs(self) -> None:
        """
        Initializes base ticker stock data
        """
        self._specs = Stock(
            ticker = self._base_ticker, 
            historical_depth = self._base_historical_depth,
            returns_freq = self._base_returns_freq
        )

        self._specs.get_stock_data()

        