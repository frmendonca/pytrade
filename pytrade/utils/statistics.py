import pandas as pd
import numpy as np

import pytrade.utils.utils as utl
from pytrade.constants import NUMERIC_ACCURACY

"""
Methods for EVT analysis
"""

def zipf_computation(data: pd.Series):
    """
    :param data: a pandas data containing the data
    :return:
    """
    if not utl.check_all_same_sign(data):
        raise ValueError("Series must be either all positive or all negative")

    data = np.sort(np.abs(data))
    survival = pd.Series([np.mean(data > x) for x in data]).clip(NUMERIC_ACCURACY, 1)

    return {"log_data": np.log(data), "log_survival": np.log(survival)}




def maximum_to_sum(data: pd.Series, k: int) -> pd.Series:
    """
    :param data: a pandas series containing the data
    :param k: an int representing the moment, k = i representing ith moment
    :return: a pandas series maximum to sum values
    """

    if not utl.check_all_same_sign(data):
        raise ValueError("Series must be either all positive or all negative")

    data = np.abs(data)
    exp_data = data**k

    partial_maximums = exp_data.cummax()
    partial_sums = exp_data.cumsum()
    return partial_maximums / partial_sums


def hill_estimator(data: pd.Series, k: int = 50):
    """
    :param data: a pandas series
    :param k: maximum order statistics to use
    :return: alpha the hill exponent
    """
    data = np.sort(data)[::-1]  # descending order
    if k >= len(data):
        raise ValueError("k must be smaller than the number of data points.")

    x_k = data[k]
    logs = np.log(data[:k]) - np.log(x_k)
    hill = np.mean(logs)
    alpha = 1 / hill
    return alpha


def mean_excess_function(
    data: pd.Series,
    thresholds: int = 100,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
):
    """
    :param data: a pandas series to analyse
    :param thresholds: number of thresholds to use
    :param min_threshold: optional, lowest threshold to check
    :param max_threshold: options, higheer threshold to check
    :return:
    """
    data = np.abs(data)
    if min_threshold is None:
        min_threshold = np.quantile(data, 0.05)
    if max_threshold is None:
        max_threshold = np.quantile(data, 0.95)

    u_vals = np.linspace(min_threshold, max_threshold, thresholds)
    mean_excess = np.zeros(len(u_vals))

    for i, u in enumerate(u_vals):
        excesses = data[data > u] - u
        if len(excesses) > 0:
            mean_excess[i] = excesses.mean()
        else:
            mean_excess[i] = np.nan

    return {"thresholds": u_vals, "mean_excesses": mean_excess}
