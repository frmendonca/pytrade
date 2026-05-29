
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from pytrade.data_models.options import OptionStrategy, RISK_FREE_RATE
from pytrade.data_models.simulation import BaseSimulationModel


# ---------------------------------------------------------------------------
# Module-level worker — must live here (not inside the class) so that
# multiprocessing can pickle it on all platforms.
# ---------------------------------------------------------------------------

def _run_single_path(args):
    """Process one simulation path and return its daily P&L array."""
    (path, strategy, starting_underlying, vol_skew, days_range, initial_cost) = args
    current_underlying = starting_underlying
    path_pnl = np.full(len(days_range), np.nan)

    for d in days_range:
        if d > 0:
            current_underlying *= (1 + path[d - 1])

        # Dynamic IV: captures the negative spot/vol correlation observed in equities.
        # When the underlying falls, IV rises; when it rises, IV falls.
        #   iv_t = iv_0 * exp(-vol_skew * log(S_t / S_0))
        # vol_skew > 0  →  negative correlation (typical for equity indices: 1.0 – 2.0).
        # vol_skew = 0  →  flat IV; identical to the original behaviour.
        if vol_skew != 0.0:
            spot_log_return = np.log(current_underlying / starting_underlying)
            dynamic_iv_factor = np.exp(-vol_skew * spot_log_return)
        else:
            dynamic_iv_factor = 1.0

        current_market_value = strategy.resolve_value(
            underlying_price=current_underlying,
            iv_factor=dynamic_iv_factor,
            days_elapsed=d
        )

        path_pnl[d] = current_market_value + initial_cost

    return path_pnl


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------

class OptionStrategySimulator(BaseSimulationModel):
    """
    Simulation engine for a single OptionStrategy.

    Inherits path-generation utilities from BaseSimulationModel and couples
    them tightly with the strategy so the caller doesn't have to pass the
    strategy or its DTE around manually.

    Usage
    -----
    sim    = OptionStrategySimulator(strategy)
    blocks = sim.generate_bootstrap_blocks(returns_data, num_resamples=10_000)
    pnl    = sim.simulate_pnl(blocks, starting_underlying=450.0, vol_skew=1.5)
    """

    def __init__(
        self,
        strategy: OptionStrategy,
        returns: np.ndarray | None = None,
        period: str = "max"
    ):
        self.strategy = strategy
        # Capture the horizon once at construction time — consistent with the
        # frozen DTE approach used by OptionModel.
        self.first_expiration: int = min(
            leg.option.days_to_expiry for leg in strategy.legs
        )

        # Validate all legs share the same underlying before fetching
        tickers = {leg.option.ticker for leg in strategy.legs}
        if len(tickers) > 1:
            raise ValueError(
                f"All strategy legs must share the same underlying. Got: {tickers}. "
                "For cross-asset strategies pass returns= explicitly."
            )

        if returns is not None:
            self.returns = returns          # user-supplied — skip network call
        else:
            self._fetch_data(tickers.pop(), period)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _fetch_data(self, ticker: str, period: str) -> None:
        """Download daily close returns from yfinance for the strategy's underlying."""
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        self.returns: np.ndarray = df["Close"].pct_change(1).dropna().to_numpy()

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------

    def generate_bootstrap_blocks(
        self,
        num_resamples: int = 1000,
        block_length: int = 10,
        seed: int | None = None,
        target_ann_vol: float | None = None
    ) -> np.ndarray:
        """
        Generate a (num_resamples, first_expiration) matrix of bootstrapped
        daily return paths using the Stationary Block Bootstrap.

        The path length and underlying return series are both derived from the
        strategy — no external data needs to be passed.

        Parameters
        ----------
        num_resamples  : Number of simulation paths.
        block_length   : Target mean block length (Geometric distribution).
                         Controls autocorrelation preservation.
                         Rule of thumb: 5–15 days for daily equity series.
        seed           : Master RNG seed for reproducibility.
        target_ann_vol : Annualized vol to rescale paths to (e.g. the ATM IV).
                         Ensures path magnitudes are consistent with current
                         option pricing. None → use historical realized vol.

        Returns
        -------
        np.ndarray of shape (num_resamples, first_expiration).
        """
        return super().generate_bootstrap_blocks(
            original_sequence=self.returns,
            seq_len=self.first_expiration,
            num_resamples=num_resamples,
            block_length=block_length,
            seed=seed,
            target_ann_vol=target_ann_vol
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_pnl(
        self,
        simulation_returns: np.ndarray,
        starting_underlying: float,
        vol_skew: float = 0.0,
        n_cores: int = -1
    ) -> np.ndarray:
        """
        Simulate P&L paths for the strategy.

        Parameters
        ----------
        simulation_returns  : 2-D array (n_simulations, n_days) of **daily**
                              underlying returns — typically the output of
                              generate_bootstrap_blocks().
        starting_underlying : Spot price at simulation start.
        iv_factor           : Baseline IV scalar on top of each leg's own IV.
                              1.0 = use market IV as-is.
        vol_skew            : Spot/vol correlation coefficient.
                              iv_t = iv_factor * exp(-vol_skew * log(S_t / S_0))
                              0.0       → flat IV (backward-compatible default).
                              1.0-2.0   → typical equity index sensitivity.
        n_cores             : Worker processes. -1 = all available CPUs.

        Returns
        -------
        np.ndarray of shape (n_simulations, first_expiration) — daily P&L per
        path, day-0 (entry day) excluded.
        """
        if n_cores == -1:
            n_cores = cpu_count()

        nsim = len(simulation_returns)
        days_range = range(0, self.first_expiration + 1)
        initial_cost = self.strategy.strategy_premium

        task_args = [
            (
                simulation_returns[j],
                self.strategy,
                starting_underlying,
                vol_skew,
                days_range,
                initial_cost,
            )
            for j in range(nsim)
        ]

        with Pool(processes=n_cores) as pool:
            results = pool.map(_run_single_path, task_args)

        return np.array(results)[:, 1:]
    

    @staticmethod
    def report_strategy_performance(strategy_result: np.ndarray, plot=True):
        """
        Pretty prints trading strategy metrics (Probability of Profit, Expected Profit, 
        VaR 1%, and Max Loss) across different time horizons.
        """
        # 1. Compute the underlying metrics
        expected_profit = strategy_result.mean(axis=0)
        prob_profit = (strategy_result > 0).mean(axis=0)

        
        q_range = [0.01, 0.05, 0.15, 0.5, 0.95]
        profit_quantiles = np.quantile(strategy_result, q_range, axis=0)
        
        # 2. Define the time steps (indices) and labels
        ndays = len(expected_profit)
        time_indices = [int(q) for q in np.linspace(int(ndays*0.1), int(ndays*0.9), 4)] + [-1]
        headers = ["Metric"] + [f"{i} Days" for i in time_indices[1:-1]] + ["Expiration"]
        
        # 3. Compile rows and format values
        rows = [
            ("Probability of Profit", [f"{prob_profit[i]:.2%}" for i in time_indices]),
            ("Expected Profit", [f"{expected_profit[i]:,.2f}" for i in time_indices]),
            ("Median", [f"{profit_quantiles[3, i]:,.2f}" for i in time_indices]),
            ("15% Percentile", [f"{profit_quantiles[2, i]:,.2f}" for i in time_indices]),
            ("5% Percentile", [f"{profit_quantiles[1, i]:,.2f}" for i in time_indices]),
            ("VaR (1%)", [f"{profit_quantiles[0, i]:,.2f}" for i in time_indices]),
            ("Max Loss", [f"{strategy_result[:, i].min():,.2f}" for i in time_indices])
        ]
        
        # 4. Print the formatted table
        row_format = "{:<25} | {:<12} | {:<12} | {:<12} | {:<12}"
        
        print("\n" + "="*83)
        print("STRATEGY PERFORMANCE SUMMARY".center(81))
        print("="*83)
        print(row_format.format(*headers))
        print("-" * 83)
        
        for metric, values in rows:
            print(row_format.format(metric, *values))
            
        print("="*83 + "\n")

        
        if plot:
            # --- Plotting ------------------------------------------------
            # Set a clean visual style
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
            
            # --- Plot 1: PnL & Quantiles ---
            # Main expected profit line
            ax[0].plot(expected_profit, color="#1a5f7a", linewidth=2.5, label="Expected Profit")
            
            # Quantile lines with a gradient or distinct dashes
            styles = [':', '--', '-.', '--']
            colors = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99"] # Distinguishable light colors for bands
            
            for i in range(1, len(q_range)):  # Excluding q=0.01 (VaR) as per original logic
                label_name = f"Quantile {int(q_range[i]*100)}%"
                ax[0].plot(profit_quantiles[i, :], color="#57606f", linestyle="--", alpha=0.6, label=label_name)
                
            ax[0].set_title("PnL Projection & Distribution Quantiles", fontsize=12, fontweight='bold', pad=10)
            ax[0].set_xlabel("Days After Origination", fontsize=10)
            ax[0].set_ylabel("PnL ($)", fontsize=10)
            ax[0].legend(frameon=True, facecolor='white', edgecolor='none')
            ax[0].grid(True, linestyle=":", alpha=0.6)

            # --- Plot 2: Probability of Profit ---
            ax[1].plot(prob_profit, color="#e63946", linewidth=2.5, label="Prob. of Profit")
            
            # Add a reference line at 50% equilibrium
            ax[1].axhline(0.5, color="#2b2d42", linestyle=":", alpha=0.5, label="50% Threshold")
            
            # Format y-axis as percentage for easier reading
            ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            ax[1].set_title("Probability of Profit Over Time", fontsize=12, fontweight='bold', pad=10)
            ax[1].set_xlabel("Days After Origination", fontsize=10)
            ax[1].set_ylabel("Probability (%)", fontsize=10)
            ax[1].legend(frameon=True, facecolor='white', edgecolor='none')
            ax[1].grid(True, linestyle=":", alpha=0.6)
            
            # Optimize spacing
            plt.tight_layout()
            plt.show()

