
import numpy as np
import pandas as pd
import numpy.typing as npt

from pytrade.data_models.base import SimulationConfig, SimulationResults
from pytrade.data_models.option import Option, OptionDirection, OptionType


def compute_hedge_cost(options: list[Option]) -> float:
    """
    Computes the total debit (credit if negative)
    for a given option strategy
    :param options: list[Option] a set of Option objects
    :returns float the debt or credit (if negative) of the strategy
    """
    total_cost = 0.0
    for option in options:
        if option.option_direction == OptionDirection.LONG:
            total_cost += 100 * option.premium * option.contracts
        elif option.option_direction == OptionDirection.SHORT:
            total_cost -= 100 * option.premium * option.contracts

    return total_cost


def compute_hedge_value(
    options: list[Option],
    returns: npt.NDArray,
    underlying: float,
    dte_change: int = 0.0,
):
    """
    Computes the value of a hedge using Black Scholes
    and assuming the opposite direction to that of the Option.
    For a SHORT call it assumes it will be bought to close
    resulting in a debit and for a LONG call it assumes
    it will be sold to close, resulting in a credit.

    :param options a list of Option objects
    :param returns an array with returns
    :param underlying a float with the current underlying asset price
    :param dte_change Optional int with the change in days_to_expiry to use in the computation. If null uses Option object dte
    :returns float value of the hedge
    """

    hedge_value = 0.0
    for option in options:
        new_iv = link_stock_returns_to_implied_volatility_exp(
            option.iv,
            returns,
            6,
            0.10
        )
        calculate = (
                100 * option.compute_black_scholes_value(
                    underlying,
                    new_iv,
                    option.days_to_expiration + dte_change
                ) * option.contracts
        )
        if option.option_direction == OptionDirection.LONG:
            hedge_value += calculate
        elif option.option_direction == OptionDirection.SHORT:
            hedge_value -= calculate

        return hedge_value



def link_stock_returns_to_implied_volatility_exp(
    current_iv: float,
    stock_return: float | npt.NDArray,
    iv_elasticity_exp: float = 5.0,  # A new default value, typically higher than linear factor
    min_iv: float = 0.01,  # Minimum implied volatility (e.g., 1%)
) -> float:
    """
    Models the relationship between underlying stock returns and option implied volatility (IV)
    using an exponential formulation. This method captures a more aggressive and non-linear
    response of IV, particularly during significant market downturns, better reflecting
    the 'leverage effect'.

    :param current_iv: The current implied volatility as a decimal (e.g., 0.20 for 20%).
                Must be greater than 0.
    :param stock_return: The percentage change in the underlying stock price as a decimal
                  (e.g., 0.01 for +1%, -0.02 for -2%).
    :param iv_elasticity_exp: A positive factor influencing the exponential change in IV.
                       Higher values mean a stronger, more non-linear response.
                       This parameter typically needs to be calibrated.
    :param min_iv: A floor for the implied volatility to ensure it never goes below a
            sensible positive value (e.g., 0.01 for 1%)..


    :returns The new implied volatility after the stock return, as a decimal.

    :raises ValueError: If current_iv is not positive, iv_elasticity_exp is negative,
                    or min_iv are invalid.
    """
    # --- Input Validation ---
    if not (0 < current_iv):
        raise ValueError(f"Current implied volatility must be positive.")
    if iv_elasticity_exp < 0:
        raise ValueError("IV elasticity factor cannot be negative.")
    if not (0 <= min_iv):
        raise ValueError(f"Min_iv ({min_iv * 100:.2f}%) must be non-negative")

    # Exponential model: current_iv * exp(-k * stock_return)
    # If stock_return is negative (drop), -iv_elasticity_exp*stock_return is positive, exp() > 1, IV increases.
    # If stock_return is positive (rise), -iv_elasticity_exp*stock_return is negative, exp() < 1, IV decreases.

    new_iv = current_iv * np.exp(-iv_elasticity_exp * stock_return)

    # Ensure IV stays within sensible bounds (min_iv and max_iv)
    return np.maximum(min_iv, new_iv)



def simulate_returns(
    *,
    returns: pd.Series,
    initial_underlying: float,
    options: list[Option],
    simulation_config: SimulationConfig
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
    :returns SimulationResults
    """

    strategy_cost = compute_hedge_cost(options)
    invested = simulation_config.portfolio_value + (
        simulation_config.contributions - strategy_cost
    )

    # Change in underlying price from T0 to T1
    underlying_next = initial_underlying * (1 + returns)
    invested_next = invested * (1 + returns)

    # Compute strategy value
    strategy_value = np.array(
        [
            compute_hedge_value(
                options,
                returns[i],
                underlying_next[i],
                simulation_config.option_dte_change,
            )
            for i in range(len(underlying_next))
        ]
    )

    # Compute total return
    final_portfolio = invested_next + strategy_value
    strategy_returns = (
        final_portfolio
        / (simulation_config.portfolio_value + simulation_config.contributions)
        - 1
    )

    return SimulationResults(
        transformed_returns=strategy_returns, original_returns=returns
    )
