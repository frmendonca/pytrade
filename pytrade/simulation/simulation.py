import typing as t
import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous_frozen as RVContinuousFrozen
from pytrade.data_models.base import SimulationConfig, SimulationResults
from pytrade.data_models.option import Option, OptionDirection


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
                100 * option._premium * option._contracts
                if option._option_direction == OptionDirection.LONG
                else 0
            )
            for option in options
        ]
    )

    credit = sum(
        [
            (
                -100 * option._premium * option._contracts
                if option._option_direction == OptionDirection.SHORT
                else 0
            )
            for option in options
        ]
    )

    return debit + credit


def compute_intrinsic_hedge_value(options: list[Option], underlying: float):
    """
    Computes the value of an hedge using intrinsic value
    and assuming oposite direction to that of the Option.
    For a SHORT call it assumes it will be bought to close
    resulting in a debit and for a LONG call it assumes it
    will be sold to close resulting in a credit.
    In all computations it uses options instrinsic value
    at expiration.

    :param options a list of Option ojects
    :returns float value of the hedge
    """
    sell_to_close = sum(
        [
            (
                100 * option.compute_intrinsic_value(underlying)
                if option._option_direction == OptionDirection.LONG
                else 0
            )
            for option in options
        ]
    )

    buy_to_close = sum(
        [
            (
                -100 * option.compute_intrinsic_value(underlying)
                if option._option_direction == OptionDirection.SHORT
                else 0
            )
            for option in options
        ]
    )

    return sell_to_close + buy_to_close


def compute_bs_hedge_value(
    options: list[Option],
    underlying: float,
    iv_change: float | None = None,
    dte_change: int | None = None,
):
    """
    Computes the value of an hedge using black scholes
    and assuming oposite direction to that of the Option.
    For a SHORT call it assumes it will be bought to close
    resulting in a debit and for a LONG call it assumes
    it will be sold to close, resulting in a credit.

    :param options a list of Option ojects
    :returns float value of the hedge
    """

    sell_to_close = sum(
        [
            (
                100
                * option.compute_black_scholes_value(
                    underlying,
                    max(0.08, option._iv * (1 + iv_change)),
                    option._days_to_expiration - dte_change,
                )
                if option._option_direction == OptionDirection.LONG
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
                    max(0.08, option._iv * (1 + iv_change)),
                    option._days_to_expiration - dte_change,
                )
                if option._option_direction == OptionDirection.SHORT
                else 0
            )
            for option in options
        ]
    )

    return sell_to_close + buy_to_close


def simulate_returns(
    *,
    generator: RVContinuousFrozen,
    initial_underlying: float,
    options: list[Option],
    simulation_config: SimulationConfig,
    volatility_generator: RVContinuousFrozen | None = None,
    compute_at_intrinsic: bool = True,
    nsim: int = 5000
) -> SimulationResults:

    hedge_cost = compute_hedge_cost(options)
    invested = simulation_config.portfolio_value + (
        simulation_config.monthly_contributions - hedge_cost
    )

    # Draw returns from generator and project next period portfolio
    returns = generator.rvs(size=nsim)
    underlying_next = initial_underlying * (1 + returns)
    invested_next = invested * (1 + returns)

    # Compute hedge value
    if compute_at_intrinsic:
        vix_returns = np.nan
        hedge_value = np.array(
            [
                compute_intrinsic_hedge_value(options, underlying)
                for underlying in underlying_next
            ]
        )
    else:
        vix_returns = volatility_generator.rvs(returns)
        hedge_value = np.array(
            [
                compute_bs_hedge_value(
                    options,
                    underlying_next[i],
                    vix_returns[i],
                    simulation_config.returns_frequency,
                )
                for i in range(len(underlying_next))
            ]
        )

    # Compute total return
    final_portfolio = invested_next + hedge_value
    hedged_returns = (
        final_portfolio
        / (simulation_config.portfolio_value + simulation_config.monthly_contributions)
        - 1
    )

    return SimulationResults(
        hedged_returns=hedged_returns, unhedged_returns=returns, vix_returns=vix_returns
    )
