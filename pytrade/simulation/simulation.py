
import numpy as np

from pytrade.simulation.base import SimulationConfig, SimulationResults
from pytrade.data_models.option import Option, OptionDirection, OptionType
from pytrade.models.vix_returns.model import model_predict as vix_predict
from pytrade.integrations.yfinance_data import YFinanceClient


def compute_hedge_cost(options: list[Option]) -> float:
    """
    Computes the total debit (credit if negative)
    for a given option strategy
    :param options: list[Option] a set  of Option objects
    :returns float the debt or credit (if negative) of the strategy
    """
    total_cost = 0.0
    for option in options:
        if option.option_direction == OptionDirection.LONG:
            total_cost += 100 * option.premium * option.quantity
        elif option.option_direction == OptionDirection.SHORT:
            total_cost -= 100 * option.premium * option.quantity

    return total_cost


def compute_hedge_value(
    options: list[Option],
    underlying: float,
    iv_change: float = 0.0,
    dte_change: int = 0.0,
):
    """
    Computes the value of a hedge using Black Scholes
    and assuming the opposite direction to that of the Option.
    For a SHORT call it assumes it will be bought to close
    resulting in a debit and for a LONG call it assumes
    it will be sold to close, resulting in a credit.

    :param options a list of Option objects
    :param underlying a float with the current underlying asset price
    :param iv_change percentage change in option iv
    :param dte_change Optional int with the change in days_to_expiry to use in the computation.
        If null uses Option object dte
    :returns float value of the hedge
    """

    hedge_value = 0.0
    for option in options:
        calculate = (
            100 * option.compute_black_scholes_value(
                underlying,
                option.iv * (1 + iv_change),
                option.days_to_expiration + dte_change
            ) * option.quantity
        )
        if option.option_direction == OptionDirection.LONG:
            hedge_value += calculate
        elif option.option_direction == OptionDirection.SHORT:
            hedge_value -= calculate

    return hedge_value


def simulate_returns(
    *,
    simulation_config: SimulationConfig,
    options: list[Option]
) -> SimulationResults:
    """
    Transforms a sequence of returns into a sequence
    of returns coupled with an option strategy.

    The transformation occurs by assuming a one step process where
    it is assumed that an option(s) is bought or sold at T0 and evaluated
    at T1. The original returns from T0 to T1 are then adjusted reflect the
    change in the option strategy

    :param simulation_config a SimulationConfig object with the configuration for the simulations
    :param options a list of Option objects with the option strategy
    :returns SimulationResults
    """

    # Setup client and fetch symbol data
    sym = simulation_config.underlying_ticker
    yf_client = YFinanceClient()
    fetcher = yf_client.fetch([sym])
    ticker = fetcher[sym]

    df = ticker.data
    df[f"{sym}_returns"] = df["Close"].pct_change(simulation_config.returns_frequency)
    df = df.dropna()

    initial_underlying = ticker.price
    returns = df[f"{sym}_returns"].dropna().values

    # Compute changes of VIX based on
    # returns of underlying
    iv_change = vix_predict(df[[f"{sym}_returns"]])

    hedge_allocation = simulation_config.hedge_allocation * simulation_config.portfolio_value
    strategy_cost = compute_hedge_cost(options)

    if strategy_cost > hedge_allocation:
        raise RuntimeError(f"Strategy cost is {strategy_cost} but maximum allocation was {hedge_allocation}. Increase hedge_allocation percentage")
    else:
        hedge_allocation = strategy_cost

    invested = simulation_config.portfolio_value - hedge_allocation

    # Change in underlying price from T0 to T1
    underlying_next = initial_underlying * (1 + returns)
    invested_next = invested * (1 + returns)

    # Compute strategy value
    strategy_value = np.array(
        [
            compute_hedge_value(
                options,
                underlying_next[i],
                iv_change[i],
                simulation_config.option_dte_change,
            )
            for i in range(len(underlying_next))
        ]
    )

    # Compute total return
    final_portfolio = np.maximum(invested_next + strategy_value, 0)
    strategy_returns = (
        final_portfolio / simulation_config.portfolio_value - 1
    )

    return SimulationResults(
        transformed_returns=strategy_returns,
        original_returns=returns,
        hedge_cost=hedge_allocation
    )



res = simulate_returns(
    simulation_config=SimulationConfig(
        underlying_ticker="SPY",
        returns_frequency=30,
        portfolio_value=66000,
        hedge_allocation=0.05,
        option_dte_change=-30
    ),
    options=[
        Option(
            ticker = "SPY",
            strike = 530,
            premium = 1.15,
            iv = 0.29883,
            r = 0.045,
            option_type=OptionType.PUT,
            option_direction=OptionDirection.LONG,
            expiration_date="2025-11-21",
            quantity=5
        ),
        Option(
            ticker = "SPY",
            strike = 500,
            premium = 0.80,
            iv = 0.34053,
            r = 0.045,
            option_type=OptionType.PUT,
            option_direction=OptionDirection.SHORT,
            expiration_date="2025-11-21",
            quantity=5
        )
    ]
)


import pandas as pd
df = pd.DataFrame([res.original_returns, res.transformed_returns], index = ["original", "transformed"]).T

np.exp(np.mean(np.log(1 + df), axis = 0)) ** 12 - 1



df.sort_values(by = "original")

