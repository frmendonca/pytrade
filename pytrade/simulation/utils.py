
import numpy as np
import yfinance as yf
from numba import njit


def compute_correlation_matrix(tickers: list[str], freq: int = 1) -> np.array:
    df = yf.download(tickers, period = "max")
    return (
        df["Close"]
        .pct_change(freq)
        .corr()
    )


def block_resample(
    original_sequence: np.array,
    block_length: int = 30,
    resample_sequence_length: int = 30,
    seed: int | None = None
) -> np.array:
    """
    Stationary Block Bootstrap (Politis & Romano, 1994).

    Block lengths are drawn from Geometric(p = 1 / block_length), so the
    *average* block length equals the supplied `block_length` but each block
    independently varies.  Wrap-around (circular) indexing ensures every
    observation is equally likely to start a block, eliminating the end-of-series
    bias of the fixed-length bootstrap.

    Parameters
    ----------
    original_sequence        : 1-D array of historical returns.
    block_length             : Target *mean* block length.
    resample_sequence_length : Desired output length
    seed                     : Optional RNG seed for reproducibility.
    """
    n = len(original_sequence)
    if block_length > n:
        raise ValueError("block_length cannot be greater than the length of the input data.")

    rng = np.random.default_rng(seed)
    p = 1.0 / block_length          # geometric distribution parameter
    result: list = []

    while len(result) < resample_sequence_length:
        actual_len = int(rng.geometric(p))          # random block length, mean = block_length
        start      = int(rng.integers(0, n))        # circular: any index equally likely
        indices    = [(start + k) % n for k in range(actual_len)]
        result.extend(original_sequence[indices].tolist())

    return np.array(result[:resample_sequence_length])



def block_resample_joint(
    sequences: list[np.ndarray],
    block_length: int,
    resample_sequence_length: int,
    seed: int | None = None
) -> list[np.ndarray]:
    """
    Stationary Joint Block Bootstrap for multiple sequences.

    Block lengths are drawn from Geometric(p = 1 / block_length), so the
    *average* block length equals the supplied `block_length` but each block
    independently varies.  Wrap-around (circular) indexing ensures every
    observation is equally likely to start a block, eliminating the end-of-series
    bias of the fixed-length bootstrap.

    Parameters
    ----------
    sequences                : list of 1-D array of historical sequences.
    block_length             : Target *mean* block length.
    resample_sequence_length : Desired output length
    seed                     : Optional RNG seed for reproducibility.
    """
    
    n = len(sequences[0])
    rng = np.random.default_rng(seed)
    p = 1.0 / block_length
    indices: list[int] = []
    while len(indices) < resample_sequence_length:
        actual_len = int(rng.geometric(p))
        start = int(rng.integers(0, n))
        indices.extend((start + k) % n for k in range(actual_len))
    indices = np.array(indices[:resample_sequence_length])
    return [seq[indices] for seq in sequences]




def compute_naked_put_return_on_margin(strike_price, underlying_price, premium):
    """
    Computes the return of a naked put underwriting strategy using standard 
    exchange margin requirements (FINRA Rule 4210 / CBOE).
    
    The margin requirement for a naked put is typically the GREATER of:
    1. 20% of underlying price - OTM amount + premium
    2. 10% of strike price + premium
    
    Returns:
        float: The return on initial margin as a decimal.
    """
    # Calculate Out-of-the-Money (OTM) amount
    # For a put, it's OTM if the underlying is above the strike
    otm_amount = max(0, underlying_price - strike_price)
    
    # Standard Margin Formula Components
    case_1 = (0.20 * underlying_price) - otm_amount + premium
    case_2 = (0.10 * strike_price) + premium
    
    margin_requirement = max(case_1, case_2)

    # Compute downside protection
    break_even = strike_price - premium
    downside_protection = break_even / underlying_price - 1
    
    # Return = Premium collected / Capital tied up
    return premium / margin_requirement, downside_protection




@njit
def apply_stop_loss(data, premium_limit):
    """
    Once a path's P&L crosses below `premium_limit`, close the position and
    forward-fill at exactly `premium_limit` for all remaining days.

    The previous implementation froze the value at the prior day's P&L, which
    could differ substantially from the stop level on gap-move days and did not
    correctly propagate the stop forward on consecutive breaches.
    """
    n, t = data.shape
    for i in range(n):
        stopped = False
        for j in range(1, t):
            if stopped:
                data[i, j] = premium_limit
            elif data[i, j] < premium_limit:
                data[i, j] = premium_limit
                stopped = True
    return data