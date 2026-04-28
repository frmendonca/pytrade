
import numpy as np

def generate_bootstrap_blocks(x: np.array, seq_len: int) -> np.array:

    # Get all possible starting points
    x_truncated = x[:-seq_len]
    n = len(x_truncated)

    return [x[i:i+seq_len] for i in range(n)]



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