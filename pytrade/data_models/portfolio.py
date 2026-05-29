
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from functools import reduce
import matplotlib.pyplot as plt
from pytrade.data_models.simulation import BaseSimulationModel, BaseSimulationResults


@dataclass
class StockPosition:
    ticker: str
    allocation: float = 1.0
    leverage: float = 1.0
    expense_ratio: float = 0.001




class Portfolio(BaseSimulationModel):

    COMPLEX_TICKER_STRUCTURES = ["?FB="]

    def __init__(self, positions: list[StockPosition]):
        self.positions = positions

        total_allocation = sum(position.allocation for position in self.positions)
        if (total_allocation > 1 + 1e-6) | (total_allocation < 1 - 1e-6):
            raise RuntimeError(f"Allocations must sum to 1.0. Got {total_allocation}")
        
        self._fetch_data() # Fetch data
        self._build_portfolio() # Construct portfolio weighted average
    

    @property
    def tickers(self):
        return [position.ticker for position in self.positions]
    
    @property
    def chained_tickers(self):
        # Unpack tickers
        chained_tickers = {}
        for ticker in self.tickers:
            for cts in self.COMPLEX_TICKER_STRUCTURES:

                # Clean up the string by replacing the indicators to make splitting uniform
                normalized_string = ticker.replace(cts, "?")
                ticker_chain = normalized_string.split("?") # Split by the '?' indicator
                chained_tickers[ticker] = ticker_chain

        return chained_tickers

    
    def _fetch_data(self):
        unpacked_chained_tickers = [t for ct in self.chained_tickers.values() for t in ct]
        # Data is extracted at daily frequency in order to apply leverage properly.
        # Leverage ETFs reset leverage daily
        df = (
            yf.download(unpacked_chained_tickers, period = "max", interval = "1d")
            .pipe(self._process_returns)
            .pipe(self._apply_leverage_factor) # Apply leverage
        )
        
        df_monthly = (1 + df).resample("ME").prod() - 1
        self.data = df_monthly


    def _process_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_returns =  (
            df["Close"]
            .pct_change(1)
        )

        for ticker, ct in self.chained_tickers.items():
            if len(ct) > 1: # is complex ticker

                data_objects = [df_returns[t] for t in ct]
                padded_df = reduce(lambda left, right: left.fillna(right), data_objects)

                df_returns[ticker] = padded_df
                df_returns = df_returns.drop(columns = [t for t in ct])

        return df_returns.dropna()    

    
    
    def _apply_leverage_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        SW = 1.1
        FFR = 0.03

        for position in self.positions:
            ticker = position.ticker
            leverage_factor = position.leverage
            expense_ratio = position.expense_ratio

            E = 0.005 * (leverage_factor - 1)
            SP = np.sign(leverage_factor) * 0.004

            cost_of_leverage = SW * (leverage_factor - 1) * (FFR + SP) + E
            df[ticker] = df[ticker] * leverage_factor - ((cost_of_leverage + expense_ratio) / 365)

        return df
    

    def _build_portfolio(self):
        allocations = [position.allocation for position in self.positions]
        self.data["PRTF"] = self.data[self.tickers].dot(allocations)


    def simulate_withdrawal_failure_rate(
        self,
        starting_portfolio: float = 10000,
        anual_withdrawal_rate: float = 0.04,
        minimum_monthly_withdrawal_amount: float = 1000,
        maximum_monthly_withdrawal_amount: float = 2000,
        horizon_years: int = 30,
        drawdown_deferral: int = 0,
        bootstrap_min_block_len: int = 1,
        bootstrap_max_block_len: int = 36,
        num_simulations: int = 1000,
        anual_inflation_rate: float = 0.03,
        seed: int = 0
    ):
        
        monthly_inflation_rate = anual_inflation_rate / 12

        sequence_lenght = horizon_years * 12
        outputs = {
            "portfolio_value": np.full(shape = (num_simulations, sequence_lenght), fill_value = np.nan),
            "sampled_returns": np.full(shape = (num_simulations, sequence_lenght), fill_value = np.nan),
            "withdrawals": np.full(shape = (num_simulations, sequence_lenght), fill_value = np.nan),
        }


        np.random.seed(seed)
        block_lens = np.random.randint(
            low = bootstrap_min_block_len,
            high = bootstrap_max_block_len,
            size = num_simulations
        )

        for i in range(num_simulations):
            b = block_lens[i]
            bootstrapped_returns = self.block_bootstrap_returns(
                original_sequence = self.data["PRTF"].values,
                block_length=b,
                resample_sequence_lenght=sequence_lenght,
                seed=seed+i
            )

            current_portfolio = starting_portfolio
            outputs["sampled_returns"][i,:] = bootstrapped_returns

            mimwa_ = minimum_monthly_withdrawal_amount
            mamwa_ = maximum_monthly_withdrawal_amount
            for t in range(sequence_lenght):
                r = bootstrapped_returns[t]
                current_portfolio *= (1 + r)

            
                monthly_withdrawal_amount = (
                    min(
                        min(
                            max(mimwa_, anual_withdrawal_rate / 12 * current_portfolio),
                            mamwa_
                        ),
                        current_portfolio
                    )
                    if t >= drawdown_deferral else 0
                )

                outputs["withdrawals"][i,t] = monthly_withdrawal_amount
                current_portfolio -= monthly_withdrawal_amount
                current_portfolio = max(0, current_portfolio)

                outputs["portfolio_value"][i,t] = current_portfolio

                # Adjust minimum_monthly_withdrawal_amount for inflation
                mimwa_ *= (1 + monthly_inflation_rate)
                mamwa_ *= (1 + monthly_inflation_rate)


        return BaseSimulationResults(simulation_output = outputs)
    


    def generate_simulation_report(self, simulation_results: BaseSimulationResults):
        import matplotlib.ticker as mticker

        dict_results  = simulation_results.simulation_output
        portfolio_arr = dict_results["portfolio_value"]   # (num_sim, seq_len)
        withdraw_arr  = dict_results["withdrawals"]        # (num_sim, seq_len)
        _, seq_len = portfolio_arr.shape
        months = np.arange(seq_len)

        # ── Derived series ─────────────────────────────────────────────────
        failure_rate    = (portfolio_arr <= 0).mean(axis=0)
        pv_q            = np.quantile(portfolio_arr, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        wd_q            = np.quantile(withdraw_arr,  [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
        terminal_values = portfolio_arr[:, -1]
        pct_depleted    = (terminal_values == 0).mean()
        survivors       = terminal_values[terminal_values > 0]
        

        # ── Shared helpers ─────────────────────────────────────────────────
        BLUE_DARK, BLUE_MID, BLUE_LIGHT = "#1a4f7a", "#4a90d9", "#c8e0f4"
        RED                             = "#c0392b"
        GREEN_DARK, GREEN_MID, GREEN_LIGHT = "#1a6b3c", "#27ae60", "#a8dfc2"

        year_ticks  = months[months % 12 == 0]
        year_labels = [f"Yr {t // 12}" for t in year_ticks]

        def apply_year_ticks(ax):
            ax.set_xticks(year_ticks)
            ax.set_xticklabels(year_labels, fontsize=8)


        def currency_fmt(x, _):
            if   x >= 1_000_000: return f"${x / 1_000_000:.1f}M"
            elif x >= 1_000:     return f"${x / 1_000:.0f}K"
            return f"${x:.0f}"

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(4, 1, figsize=(12, 14))
        fig.suptitle("Portfolio Simulation Report", fontsize=15, fontweight="bold")

        # ── 1. Depletion Rate ──────────────────────────────────────────────
        ax = axes[0]
        ax.plot(months, failure_rate * 100, color=RED, linewidth=2)
        for ref, ls in [(5, "--"), (10, ":")]:
            ax.axhline(ref, color="grey", linewidth=0.9, linestyle=ls, alpha=0.7, label=f"{ref}%")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.tick_params(axis="x", labelrotation=65)
        ax.set_title("Portfolio Depletion Rate", fontweight="bold")
        ax.set_ylabel("Cumulative probability")
        ax.legend(title="Reference", fontsize=8)
        apply_year_ticks(ax)

        # ── 2. Portfolio Value Fan ─────────────────────────────────────────
        ax = axes[1]
        ax.fill_between(months, pv_q[0], pv_q[4], color=BLUE_LIGHT, alpha=0.55, label="5th – 95th")
        ax.fill_between(months, pv_q[1], pv_q[3], color=BLUE_MID,   alpha=0.50, label="25th – 75th")
        ax.plot(months, pv_q[2], color=BLUE_DARK, linewidth=2, label="Median")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_fmt))
        ax.tick_params(axis="x", labelrotation=65)
        ax.set_title("Portfolio Value", fontweight="bold")
        ax.set_ylabel("Value")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        apply_year_ticks(ax)

        # ── 3. Terminal Value Histogram ────────────────────────────────────
        ax = axes[2]
        if len(survivors) > 0:
            ax.hist(survivors, bins=40, color=BLUE_MID, edgecolor="white",
                    linewidth=0.5, alpha=0.85, label="Survivors")
            ax.axvline(np.median(survivors), color=BLUE_DARK, linewidth=1.5,
                       linestyle="--", label=f"Median: {currency_fmt(np.median(survivors), None)}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(currency_fmt))
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_title("Terminal Portfolio Value", fontweight="bold")
        ax.set_xlabel(f"Value at year {seq_len // 12}")
        ax.set_ylabel("Frequency")
        ax.annotate(f"{pct_depleted:.1%} fully depleted", xy=(0.97, 0.92),
                    xycoords="axes fraction", ha="right", fontsize=10,
                    color=RED, fontweight="bold")
        ax.legend(fontsize=8)

        # ── 4. Withdrawal Trajectory ───────────────────────────────────────
        ax = axes[3]
        ax.fill_between(months, wd_q[0], wd_q[4], color=GREEN_LIGHT, alpha=0.55, label="10th – 90th")
        ax.fill_between(months, wd_q[1], wd_q[3], color=GREEN_MID,   alpha=0.50, label="25th – 75th")
        ax.plot(months, wd_q[2], color=GREEN_DARK, linewidth=2, label="Median")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(currency_fmt))
        ax.tick_params(axis="x", labelrotation=65)
        ax.set_title("Monthly Withdrawal", fontweight="bold")
        ax.set_ylabel("Amount")
        ax.legend(fontsize=8)
        apply_year_ticks(ax)

        plt.tight_layout()
        plt.show()



    def plot_simulation_path(
        self,
        simulation_results: BaseSimulationResults,
        rank: int
    ) -> None:
        """
        Plot portfolio value, return sequence, and withdrawal amount for a single
        simulation path selected by rank.

        Paths are ranked primarily by final portfolio value (ascending). Among paths
        that depleted to zero, the secondary sort is by the first month the portfolio
        hit zero (ascending), so the fastest failures rank lowest.

        Parameters
        ----------
        simulation_results : BaseSimulationResults
            Output of simulate_withdrawal_failure_rate.
        rank : int
            0-based rank (worst → best).
            rank=0 → worst path, rank=num_simulations-1 → best path.
        """
        dict_results    = simulation_results.simulation_output
        portfolio_arr   = dict_results["portfolio_value"]   # (num_simulations, seq_len)
        num_simulations = portfolio_arr.shape[0]
        seq_len         = portfolio_arr.shape[1]

        # --- Ranking -------------------------------------------------------
        final_values = portfolio_arr[:, -1]

        # For each path: first month the portfolio hit zero; seq_len if it never did.
        hit_zero  = (portfolio_arr == 0).any(axis=1)
        first_zero = np.where(
            hit_zero,
            np.argmax(portfolio_arr == 0, axis=1),
            seq_len  # never depleted → treated as "last" in secondary sort (irrelevant)
        )

        # lexsort: rightmost key = primary sort, leftmost = secondary sort for ties.
        # Primary: final value ascending; Secondary: first-zero month ascending.
        ranked_indices = np.lexsort((first_zero, final_values))
        sim_idx        = ranked_indices[rank]

        # --- Extract path data ---------------------------------------------
        portfolio_values = portfolio_arr[sim_idx, :]
        returns          = dict_results["sampled_returns"][sim_idx, :]
        withdrawals      = dict_results["withdrawals"][sim_idx, :]
        months           = range(seq_len)

        # --- Build title ---------------------------------------------------
        title = f"Path Analysis — Rank {rank} of {num_simulations}  |  Final Value: {final_values[sim_idx]:,.0f}"
        if hit_zero[sim_idx]:
            title += f"  |  Depleted at month {first_zero[sim_idx]}"

        # --- Plot ----------------------------------------------------------
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Portfolio value
        axes[0].plot(months, portfolio_values, color="darkblue")
        axes[0].set_xlabel("Months drawing down")
        axes[0].set_ylabel("Portfolio Value")
        axes[0].set_title("Portfolio Value")
        axes[0].grid(alpha=0.25)

        # Monthly return sequence
        axes[1].plot(months, returns, color="darkgreen")
        axes[1].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("Months drawing down")
        axes[1].set_ylabel("Monthly Return")
        axes[1].set_title("Return Sequence")
        axes[1].grid(alpha=0.25)

        # Withdrawal amounts
        axes[2].plot(months, withdrawals, color="darkred")
        axes[2].set_xlabel("Months drawing down")
        axes[2].set_ylabel("Withdrawal Amount")
        axes[2].set_title("Withdrawal Amount")
        axes[2].grid(alpha=0.25)

        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        plt.show()
