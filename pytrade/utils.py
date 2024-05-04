
from copy import deepcopy
import datetime
import math
import numpy as np
import pandas as pd
from typing import Iterable, List
from pytrade.data_models.options import Option


def timestamp():
    return datetime.datetime.today().strftime("%Y%m%d_%HH%MM%SS")

def return_frequency_to_minor(return_frequency: int) -> int:
    return math.floor(360/return_frequency)

def compute_compound_returns(sequence: Iterable) -> Iterable:
    return np.prod(1 + sequence, axis = 1)

def compute_statistics(sequence: Iterable, n: int) -> dict[str, float]:
    """
    This method computes a set of statistics for the underlying simulation
    :param sequence: Iterable the simulated return sequence
    :param n: int the number of periods to compute the statistics
    :returns dict[str, float] a dict with the statistics calculated
    """
    return {
            'CAGR - Median': np.median(sequence**(1/(n)) - 1),
            'CAGR - 5th percentile': np.quantile(sequence**(1/(n)) - 1, 0.05),
            'Probability of negative CAGR': np.mean(sequence < 1)
        }


def compute_future_underlying(current_underlying: float, returns: np.array) -> np.array:
    """
    This method computes the future value of the underlying asset
    :param current_underlying: float the current value of the underlying asset
    :param returns: np.array the array with returns
    :returns np.array with future underlying asset values
    """
    return current_underlying*(1 + returns)


def beta_weight_return_distribution(return_sample: pd.Series, portfolio_beta: float = 1.0) -> np.array:
    """
    This method modifies the return distribution given portfolio beta
    :param return_sample: pd.Series the return distribution
    :param portfolio_beta: float is the portfolio beta
    :return np.array with beta weighted returns
    """
    return np.random.choice(return_sample, size = len(return_sample), replace = False)*portfolio_beta


def compute_hedge_allocation(
        portfolio_market_value: float,
        nb_option_contracts: int,
        option_premium: float
) -> float:
    
    """
    This method computes the cost and % allocation of the hedge
    :param portfolio_market_value: float is the market value of the portfolio
    :param nb_option_contracts: int is the number of option contracts
    :param option_premium: float is the premium for each option contract
    :return hedge allocation
    """

    hedge_cost = 100*nb_option_contracts*option_premium
    hedge_allocation = hedge_cost/portfolio_market_value
    
    return hedge_allocation
    

def compute_hedge_return(
        option: Option | List[Option],
        underlying_future: np.array
) -> np.array:
    
    """
    This method computes the return of the hedge
    :param option: Option | List[Option] the option contract class or list of option contracts
    :param underlying_future: np.array the array with future underlying values
    :return hedge_return
    """

    if isinstance(option, list):
        linked_opt_to_price = [(option[i], underlying_future[i]) for i in range(len(option))]
        hedge_future_value = np.array([x[0].compute_black_scholes_price(x[1]) for x in linked_opt_to_price])
        hedge_return = (hedge_future_value - option[0].premium)/option[0].premium
    else:
        hedge_future_value = np.array([option.compute_black_scholes_price(s) for s in underlying_future])
        hedge_return = (hedge_future_value - option.premium)/option.premium
    
    return hedge_return


def simulate_iv_change(
        return_sample: np.array,
        option: Option
) -> List[Option]:
    """
    This method takes in the option contract and the return sequence and simulates the IV change for this contract
    The result is a multiplication of the option: Option object where each one has different IV
    :param return_sample: np.array the return sequence
    :param option: Option - the option contract
    :return List[Option] one new option contract with simulated IV
    """
    # Simulate VIX returns based on stock returns
    vix_returns = simulate_vix_returns(return_sample)

    # Apply these to option IV
    new_option_iv = option.iv*(1 + vix_returns)

    # Generate list of option contracts that are replicas of the original option contract
    #   where we change iv
    list_options = []
    for new_iv in new_option_iv:
        new_option = deepcopy(option)
        new_option.iv = new_iv
        list_options.append(new_option)
    
    return list_options



def simulate_vix_returns(return_sample: np.array) -> np.array:
    """
    This method simulates the VIX returns based on historical observations of stock market returns
    :param return_sample: np.array the return sequence
    :return an array with VIX returns
    """
    def conditional_gaussian(el: float) -> float:
        if el <= -0.25:
            return np.random.normal(loc = 2.505734, scale = 0.834054, size = 1)
        elif el <= -0.20:
            return np.random.normal(loc = 2.505734, scale = 0.834054, size = 1)
        elif el <= -0.15: 
            return np.random.normal(loc = 0.715229	, scale = 0.655405, size = 1)
        elif el <=  -0.1:   
            return np.random.normal(loc = 0.561455	, scale = 0.382293, size = 1)
        elif el <= -0.05:   
            return np.random.normal(loc = 0.342596	, scale = 0.356637, size = 1)
        elif el <= 0.0:	    
            return np.random.normal(loc = 0.123496	, scale = 0.224343, size = 1)
        elif el <= 0.05:    
            return np.random.normal(loc = -0.038899, scale = 0.160567, size = 1)
        elif el <= 0.1:     
            return np.random.normal(loc = -0.147999, scale = 0.150062, size = 1)
        elif el <= 0.15:    
            return np.random.normal(loc = -0.231449, scale = 0.143455, size = 1)
        else:
            return np.random.normal(loc = -0.371148, scale = 0.133988, size = 1)


    draw_vectorized = np.vectorize(conditional_gaussian)
    return draw_vectorized(return_sample)