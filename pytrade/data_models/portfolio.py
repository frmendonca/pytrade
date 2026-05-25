
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
        for position in self.positions:
            ticker = position.ticker
            leverage_factor = position.leverage
            df[ticker] = df[ticker] * leverage_factor

        return df
    

    def _build_portfolio(self):
        allocations = [position.allocation for position in self.positions]
        self.data["PRTF"] = self.data[self.tickers].dot(allocations)


    def simulate_withdrawal_failure_rate(
        self,
        starting_portfolio: float = 10000,
        anual_withdrawal_rate: float = 0.04,
        minimum_monthly_withdrawal_amount: float = 1000,
        horizon_years: int = 30,
        drawdown_deferral: int = 0,
        bootstrap_min_block_len: int = 1,
        bootstrap_max_block_len: int = 36,
        num_simulations: int = 1000,
        seed: int = 0
    ):
        
        sequence_lenght = horizon_years * 12
        outputs = {
            "portfolio_value": np.full(shape = (num_simulations, sequence_lenght), fill_value = np.nan),
            "sampled_returns": np.full(shape = (num_simulations, sequence_lenght), fill_value = np.nan),
            "withdrawals": np.full(shape = (num_simulations, sequence_lenght), fill_value = np.nan),
        }


        np.random.seed(seed)
        block_lens = np.random.randint(low = bootstrap_min_block_len, high = bootstrap_max_block_len, size = num_simulations)
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

            for t in range(sequence_lenght):
                r = bootstrapped_returns[t]
                current_portfolio *= (1 + r)

            
                monthly_withdrawal_amount = (
                    min(max(minimum_monthly_withdrawal_amount, anual_withdrawal_rate / 12 * current_portfolio), current_portfolio)
                    if t >= drawdown_deferral else 0
                )

                outputs["withdrawals"][i,t] = monthly_withdrawal_amount
                current_portfolio -= monthly_withdrawal_amount
                current_portfolio = max(0, current_portfolio)

                outputs["portfolio_value"][i,t] = current_portfolio


        return BaseSimulationResults(simulation_output = outputs)
    


    def generate_simulation_report(self, simulation_results: BaseSimulationResults):

        dict_results = simulation_results.simulation_output
        starting_portfolio_aprox = np.median(dict_results["portfolio_value"][:,0])

        failure_rate = (dict_results["portfolio_value"] <= 0).mean(axis = 0)
        portfolio_value_ci = np.quantile(dict_results["portfolio_value"], q = [0.025, 0.975], axis = 0).T
        avg_portfolio_value = dict_results["portfolio_value"].mean(axis = 0)
        prob_amount_falling_below = (dict_results["portfolio_value"] <= starting_portfolio_aprox).mean(axis = 0)
        withdrawals = np.quantile(dict_results["withdrawals"], q = [0.001, 0.01, 0.025, 0.05, 0.10], axis = 0).T
        value_at_risk = np.quantile(dict_results["portfolio_value"], q = [0.001, 0.01, 0.025, 0.05], axis = 0).T


        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Failure rate
        fig, ax = plt.subplots(1,1,figsize = (8, 6))
        ax.plot(failure_rate, color = "darkred")
        ax.grid(alpha = 0.25)
        ax.set_xlabel("Months drawing down")
        ax.set_ylabel("Failure rate")
        fig.suptitle("Failure rate")
        plt.tight_layout()
        plt.show()

        # Portfolio value
        fig, ax = plt.subplots(1,1,figsize = (8, 6))
        ax.plot(avg_portfolio_value, color = "darkred", label = "Average")
        ax.fill_between(x = range(portfolio_value_ci.shape[0]), y1 = portfolio_value_ci[:,0], y2 = portfolio_value_ci[:,1], color = "darkblue", linestyle = "--", alpha = 0.1)
        ax.grid(alpha = 0.25)
        ax.set_xlabel("Months drawing down")
        ax.set_ylabel("Value")
        fig.suptitle("Portfolio Value")
        plt.tight_layout()
        plt.show()

        # Prob falling short
        fig, ax = plt.subplots(1,1,figsize = (8, 6))
        ax.plot(prob_amount_falling_below)
        ax.grid(alpha = 0.25)
        ax.set_xlabel("Months drawing down")
        ax.set_ylabel("Probability")
        fig.suptitle("Probability of portfolio value falling below starting point")
        plt.tight_layout()
        plt.show()


        # Withdrawals
        fig, ax = plt.subplots(1,1,figsize = (8, 6))
        ax.plot(withdrawals)
        ax.grid(alpha = 0.25)
        ax.set_xlabel("Months drawing down")
        ax.set_ylabel("Withdrawal amount")
        ax.legend(title = "Percentile", labels = ["0.1%", "1%", "2.5%", "5%", "10%"])
        fig.suptitle("Withdrawal Amount - Selected Percentiles")
        plt.tight_layout()
        plt.show()


        # VaR
        fig, ax = plt.subplots(1,1,figsize = (8, 6))
        ax.plot(value_at_risk)
        ax.grid(alpha = 0.25)
        ax.set_xlabel("Months drawing down")
        ax.set_ylabel("Portfolio Value")
        ax.legend(title = "Percentile", labels = ["0.1%", "1%", "2.5%", "5%"])
        fig.suptitle("Portfolio Value - Selected Percentiles")
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
