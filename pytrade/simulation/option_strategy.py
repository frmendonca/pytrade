
import numpy as np
from multiprocessing import Pool, cpu_count
from pytrade.data_models.models import OptionStrategy

def _run_single_path(args):
    """Helper function to process a single simulation path."""
    (path, strategy, starting_underlying, iv_factor, days_range, initial_cost) = args
    current_underlying = starting_underlying
    path_pnl = np.full(len(days_range), np.nan)

    for d in days_range:
        # Update underlying price based on returns
        if d > 0:
            current_underlying *= (1 + path[d-1])

        # Market value of the position at current time
        # We use strategy_value because resolve_value returns 
        # position value when net long and negative if net short
        current_market_value = strategy.resolve_value(
            underlying_price = current_underlying,
            iv_factor = iv_factor,
            days_elapsed = d
        )

        path_pnl[d] = current_market_value + initial_cost

    return path_pnl


def simulate_pnl(
    strategy: OptionStrategy,
    simulation_returns: np.array,
    starting_underlying: float,
    iv_factor: float = 1.0,
    n_cores = -1
):

    # Determine number of workers
    if n_cores == -1:
        n_cores = cpu_count()

    # Simulations Parameters
    nsim = len(simulation_returns)

    # Simulate paths and check Strategy PnL over DTE
    first_dte = min([opt.days_to_expiry for opt in strategy.options])
    days_range = range(0, int(first_dte) + 1) # Ensure we include expiration date
    
    # The cost basis (negative means net debit/paid out, positive means net credit)
    initial_cost = strategy.strategy_premium

    # Prepare arguments for the worker pool
    # We pass the strategy and parameters to every worker
    task_args = [
        (simulation_returns[j], strategy, starting_underlying, iv_factor, days_range, initial_cost)
        for j in range(nsim)
    ]

    # Execute in parallel
    with Pool(processes=n_cores) as pool:
        # map preserves order, ensuring strategy_pnl[j] corresponds to simulation_returns[j]
        results = pool.map(_run_single_path, task_args)
    
    return np.array(results)