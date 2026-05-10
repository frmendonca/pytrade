
import numpy as np
import yfinance as yf


def compute_correlation_matrix(tickers: list[str], freq: int = 1) -> np.array:
    df = yf.download(tickers, period = "max")
    return (
        df["Close"]
        .pct_change(freq)
        .corr()
    )


def generate_bootstrap_blocks(x: np.array, seq_len: int, num_resamples: int = 1000) -> np.array:

    n = len(x)
    if seq_len > n:
        raise ValueError("seq_len cannot be greater than the length of the input data.")
    
    max_start_idx = n - seq_len + 1
    start_indices = np.random.randint(0, max_start_idx, size = num_resamples)

    return np.array([x[i : i + seq_len] for i in start_indices])



def block_resample(original_sequence: np.array, block_length: int = 30, resample_sequence_lenght: int = 30) -> np.array:

    n = len(original_sequence)
    if block_length > n:
        raise ValueError("block_length cannot be greater than the length of the input data.")
    
    number_of_blocks = resample_sequence_lenght // block_length + 1
    max_start_idx = n - block_length + 1
    
    # Sample blocks
    sampled_start_idx = np.random.choice(max_start_idx, number_of_blocks, replace = True)
    return np.hstack([original_sequence[i: i + block_length] for i in sampled_start_idx])




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