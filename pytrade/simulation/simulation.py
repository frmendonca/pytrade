
import numpy as np
import pandas as pd

from pytrade.data_models.base import SimulationConfig, SimulationResults
from pytrade.data_models.option import Option, OptionDirection, OptionType


def compute_hedge_cost(options: list[Option]) -> float:
    """
    Computes the total debit (credit if negative)
    for a given option strategy
    :param options: list[Option] a set of Option objects
    :returns float the debt or credit (if negative) of the strategy
    """

    debit = sum(
        [
            (
                100 * option.premium * option.contracts
                if option.option_direction == OptionDirection.LONG
                else 0
            )
            for option in options
        ]
    )

    credit = sum(
        [
            (
                -100 * option.premium * option.contracts
                if option.option_direction == OptionDirection.SHORT
                else 0
            )
            for option in options
        ]
    )

    return debit + credit


def compute_intrinsic_hedge_value(options: list[Option], underlying: float) -> float:
    """
    Computes the value of an hedge using intrinsic value
    and assuming oposite direction to that of the Option.
    For a SHORT call it assumes it will be bought to close
    resulting in a debit and for a LONG call it assumes it
    will be sold to close resulting in a credit.
    In all computations it uses options instrinsic value
    at expiration.

    :param options a list of Option ojects
    :param underlying a float with the current underlying price
    :returns float value of the hedge
    """
    sell_to_close = sum(
        [
            (
                100 * option.compute_intrinsic_value(underlying) * option.contracts
                if option.option_direction == OptionDirection.LONG
                else 0
            )
            for option in options
        ]
    )

    buy_to_close = sum(
        [
            (
                -100 * option.compute_intrinsic_value(underlying) * option.contracts
                if option.option_direction == OptionDirection.SHORT
                else 0
            )
            for option in options
        ]
    )

    return sell_to_close + buy_to_close


def compute_bs_hedge_value(
    options: list[Option],
    underlying: float,
    iv_change: float = 0.0,
    dte_change: int = 0.0,
):
    """
    Computes the value of a hedge using black scholes
    and assuming oposite direction to that of the Option.
    For a SHORT call it assumes it will be bought to close
    resulting in a debit and for a LONG call it assumes
    it will be sold to close, resulting in a credit.

    :param options a list of Option ojects
    :param underlying a float with the current underlying asset price
    :param iv_change Optional float with the change in implied volatily. If null use Option object IV
    :param dte_change Optional int with the change in days_to_expiry to use in the computation. If null uses Option object dte
    :returns float value of the hedge
    """

    sell_to_close = sum(
        [
            (
                100
                * option.compute_black_scholes_value(
                    underlying,
                    option.iv + iv_change,
                    option.days_to_expiration - dte_change,
                ) * option.contracts
                if option.option_direction == OptionDirection.LONG
                else 0
            )
            for option in options
        ]
    )

    buy_to_close = sum(
        [
            (
                -100
                * option.compute_black_scholes_value(
                    underlying,
                    option.iv + iv_change,
                    option.days_to_expiration - dte_change,
                ) * option.contracts
                if option.option_direction == OptionDirection.SHORT
                else 0
            )
            for option in options
        ]
    )

    return sell_to_close + buy_to_close


def simulate_returns(
    *,
    returns: pd.Series,
    initial_underlying: float,
    options: list[Option],
    simulation_config: SimulationConfig,
    compute_at_intrinsic: bool = True,
) -> SimulationResults:
    """
    Transforms a sequence of returns into a sequence
    of returns coupled with an option strategy.

    The transformation occurs by assuming a one step process where
    it is assumed that an option(s) is bought or sold at T0 and evaluated
    at T1. The original returns from T0 to T1 are then adjusted reflect the
    change in the option strategy

    :param returns a pandas series containing original returns
    :param initial_underlying a float with the current underlying price
    :param options a list of Option objects with the option strategy
    :param simulation_config a SimulationConfig object with the configuration for the simulations
    :param compute_at_intrinsic boolean equal to True if options should be evaluated at expiration
    :returns SimulationResults
    """

    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)

    strategy_cost = compute_hedge_cost(options)
    invested = simulation_config.portfolio_value + (
        simulation_config.monthly_contributions - strategy_cost
    )

    # Change in underlying price from T0 to T1
    underlying_next = initial_underlying * (1 + returns)
    invested_next = invested * (1 + returns)

    # Compute strategy value
    if compute_at_intrinsic:
        strategy_value = np.array(
            [
                compute_intrinsic_hedge_value(options, underlying)
                for underlying in underlying_next
            ]
        )
    else:
        strategy_value = np.array(
            [
                compute_bs_hedge_value(
                    options,
                    undnext,
                    0, #TODO: Add change in IV according to market return
                    simulation_config.option_dte_change,
                )
                for undnext in underlying_next
            ]
        )

    # Compute total return
    final_portfolio = invested_next + strategy_value
    strategy_returns = (
        final_portfolio
        / (simulation_config.portfolio_value + simulation_config.monthly_contributions)
        - 1
    )

    return SimulationResults(
        transformed_returns=strategy_returns, original_returns=returns
    )
