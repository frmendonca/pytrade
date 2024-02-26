
import datetime
import math
import numpy as np
from typing import Iterable


def timestamp():
    return datetime.datetime.today().strftime("%Y%m%d_%HH%MM%SS")

def return_frequency_to_minor(return_frequency: int) -> int:
    return math.floor(360/return_frequency)

def compute_compound_returns(sequence: Iterable) -> Iterable:
    return np.prod(1 + sequence, axis = 1)

def compute_statistics(sequence: Iterable, n:int) -> dict[str, float]:
    return {
            'CAGR - Median': np.median(sequence**(1/(n)) - 1),
            'CAGR - 5th percentile': np.quantile(sequence**(1/(n)) - 1, 0.05),
            'Probability of negative CAGR': np.mean(sequence < 1)
        }


