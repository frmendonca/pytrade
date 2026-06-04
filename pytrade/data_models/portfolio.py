
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
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
        
        self._fetch_data()               # Fetch data
        self._apply_external_padding()   # Pad historical returns from local parquet files
        self._build_portfolio()          # Construct portfolio weighted average
    

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
        
    
        df_monthly = (1 + df).resample("ME").prod(min_count=1) - 1
        self.data = df_monthly

        self._fetch_inflation_data() # Fetches inflation from ECB's ReST API


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

        return df_returns

    
    
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
    

    def _fetch_inflation_data(self):
        # --- ECB (recent, ~1997+) -----------------------------------------
        url = (
            "https://data-api.ecb.europa.eu/service/data/"
            "ICP/M.PT.N.000000.4.ANR?format=jsondata"
        )

        resp  = requests.get(url, timeout=15)
        resp.raise_for_status()
        data  = resp.json()

        observations = data["dataSets"][0]["series"]["0:0:0:0:0:0"]["observations"]
        dates_raw    = data["structure"]["dimensions"]["observation"][0]["values"]

        dates  = pd.to_datetime([d["id"] for d in dates_raw], format="%Y-%m") + pd.offsets.MonthEnd(0)
        values = [observations[str(i)][0] for i in range(len(dates))]

        hicp = pd.Series(values, index=dates, dtype=float)
        # ECB reports annual % change -> convert to equivalent monthly rate
        monthly_inflation = (1 + hicp / 100) ** (1 / 12) - 1

        # --- Local parquet backfill (historical, e.g. pre-1997) -----------
        external_dir = Path(__file__).resolve().parent.parent.parent / "data" / "external"
        if external_dir.exists():
            for parquet_path in external_dir.glob("*.parquet"):
                ext_df = pd.read_parquet(parquet_path)
                if "INFLATION" not in ext_df.columns:
                    continue
                ext_df.index = ext_df.index + pd.offsets.MonthEnd(0)
                # ECB wins where it has data; parquet fills historical gaps
                monthly_inflation = monthly_inflation.combine_first(ext_df["INFLATION"])
                break  # first matching file wins

        # --- Align to self.data index -------------------------------------
        aligned = monthly_inflation.reindex(self.data.index).ffill()
        if aligned.notna().any():
            self.data["INFLATION"] = aligned


    def _apply_external_padding(self):
        
        external_dir = Path(__file__).resolve().parent.parent.parent / "data" / "external"
        if not external_dir.exists():
            return

        portfolio_tickers = set(self.tickers)

        for parquet_path in external_dir.glob("*.parquet"):
            ext_df = pd.read_parquet(parquet_path)
            
            if not isinstance(ext_df.index, pd.DatetimeIndex):
                continue

            # Normalize index to month-end to align with self.data
            ext_df.index = ext_df.index + pd.offsets.MonthEnd(0)

            matching_tickers = portfolio_tickers & set(ext_df.columns)
            if not matching_tickers:
                continue

            # Extend self.data to cover the historical date range in the parquet file
            combined_index = self.data.index.union(ext_df.index)
            self.data = self.data.reindex(combined_index)
            
            for ticker in matching_tickers:
                # yfinance data takes priority; parquet fills in historical gaps
                self.data[ticker] = self.data[ticker].fillna(
                    ext_df[ticker].reindex(combined_index)
                )


    def _build_portfolio(self):
        allocations = [position.allocation for position in self.positions]
        self.data["PRTF"] = self.data[self.tickers].dot(allocations)
        self.data = self.data.dropna(subset=["PRTF"])


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
        inflation_rate_fallback: float = 0.03,
        seed: int = 0
    ):
        
        has_empirical_inflation = (
            "INFLATION" in self.data.columns and self.data["INFLATION"].notna().any()
        )

        sequence_length = horizon_years * 12
        outputs = {
            "portfolio_value": np.full(shape = (num_simulations, sequence_length), fill_value = np.nan),
            "sampled_returns": np.full(shape = (num_simulations, sequence_length), fill_value = np.nan),
            "sampled_inflation": np.full(shape = (num_simulations, sequence_length), fill_value = np.nan),
            "withdrawals": np.full(shape = (num_simulations, sequence_length), fill_value = np.nan),
        }


        np.random.seed(seed)
        block_lens = np.random.randint(
            low = bootstrap_min_block_len,
            high = bootstrap_max_block_len,
            size = num_simulations
        )

        for i in range(num_simulations):
            b = block_lens[i]
            if has_empirical_inflation:
                bootstrapped_returns, bootstrapped_inflation = self.block_resample_joint(
                    sequences=[self.data["PRTF"].values, self.data["INFLATION"].values],
                    block_length=b,
                    resample_sequence_length=sequence_length,
                    seed=seed + i
                )
            else:
                bootstrapped_returns = self.block_bootstrap_returns(
                    original_sequence=self.data["PRTF"].values,
                    block_length=b,
                    resample_sequence_length=sequence_length,
                    seed=seed + i
                )
                bootstrapped_inflation = np.full(sequence_length, inflation_rate_fallback / 12)


            current_portfolio = starting_portfolio
            outputs["sampled_returns"][i,:] = bootstrapped_returns
            outputs["sampled_inflation"][i,:] = bootstrapped_inflation

            mimwa_ = minimum_monthly_withdrawal_amount
            mamwa_ = maximum_monthly_withdrawal_amount
            for t in range(sequence_length):
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
                mimwa_ *= (1 + bootstrapped_inflation[t])
                mamwa_ *= (1 + bootstrapped_inflation[t])


        return BaseSimulationResults(simulation_output = outputs)


    def estimate_perpetual_withdrawal_rate(
        self,
        horizon_years: int = 30,
        num_simulations: int = 1000,
        bootstrap_min_block_len: int = 1,
        bootstrap_max_block_len: int = 36,
        inflation_rate_fallback: float = 0.03,
        seed: int = 0
    ) -> BaseSimulationResults:
        """
        For each bootstrapped path, solve analytically for the annual withdrawal rate w*
        such that the final portfolio value equals the inflation-adjusted initial value
        (i.e. the portfolio maintained its real purchasing power over the horizon).

        Returns a distribution of w* across simulations. Negative values indicate paths
        where returns were so poor that real value could not be preserved even at w=0.
        """
        has_empirical_inflation = (
            "INFLATION" in self.data.columns and self.data["INFLATION"].notna().any()
        )

        T = horizon_years * 12
        prtf_seq = self.data["PRTF"].values
        inf_seq  = self.data["INFLATION"].values if has_empirical_inflation else None

        w_stars     = np.empty(num_simulations)
        ret_store   = np.empty((num_simulations, T))
        inf_store   = np.empty((num_simulations, T))

        np.random.seed(seed)
        block_lens = np.random.randint(
            low=bootstrap_min_block_len,
            high=bootstrap_max_block_len,
            size=num_simulations
        )

        for i in range(num_simulations):
            b = block_lens[i]
            if has_empirical_inflation:
                r, inf = self.block_resample_joint(
                    sequences=[prtf_seq, inf_seq],
                    block_length=b,
                    resample_sequence_length=T,
                    seed=seed + i
                )
            else:
                r   = self.block_bootstrap_returns(prtf_seq, block_length=b,
                                                   resample_sequence_length=T, seed=seed + i)
                inf = np.full(T, inflation_rate_fallback / 12)

            # CPI_{t-1} for each withdrawal month t=1..T  →  [1, 1+inf[0], ...]
            cpi_mult = np.concatenate([[1.0], np.cumprod(1 + inf[:-1])])

            # Suffix growth: suffix_g[k] = prod(1+r[j] for j=k..T-1), suffix_g[T] = 1
            suffix_g = np.concatenate([np.cumprod((1 + r)[::-1])[::-1], [1.0]])

            total_growth = suffix_g[0]        # A
            terminal_cpi = np.prod(1 + inf)   # CPI_T
            B = np.dot(cpi_mult, suffix_g[1:]) / 12.0

            w_stars[i]   = (total_growth - terminal_cpi) / B
            ret_store[i] = r
            inf_store[i] = inf

        return BaseSimulationResults(simulation_output={
            "perpetual_withdrawal_rates": w_stars,
            "sampled_returns":            ret_store,
            "sampled_inflation":          inf_store,
        })


    def plot_perpetual_withdrawal_rate_distribution(
        self,
        simulation_results: BaseSimulationResults
    ) -> None:
        import matplotlib.ticker as mticker

        outputs  = simulation_results.simulation_output
        w_stars  = outputs["perpetual_withdrawal_rates"]
        ret_store = outputs["sampled_returns"]

        percentiles = [1, 5, 15, 50, 95]
        pct_values  = np.percentile(w_stars, percentiles)
        total_growth_per_sim = np.prod(1 + ret_store, axis=1)

        BLUE_DARK, BLUE_MID = "#1a4f7a", "#4a90d9"
        RED                  = "#c0392b"

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Perpetual Withdrawal Rate Distribution", fontsize=14, fontweight="bold")

        # ── 1. Histogram with percentile markers ──────────────────────────
        ax = axes[0]
        ax.hist(w_stars, bins=60, color=BLUE_MID, edgecolor="white", linewidth=0.4, alpha=0.85)
        colors = [RED, "#e67e22", BLUE_DARK, "#27ae60", "#1a6b3c"]
        for pct, val, col in zip(percentiles, pct_values, colors):
            ax.axvline(val, color=col, linewidth=1.5, linestyle="--",
                       label=f"p{pct}: {val:.1%}")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_xlabel("Annual withdrawal rate (w*)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Perpetual Withdrawal Rate")
        ax.legend(fontsize=8)

        # ── 2. Scatter: w* vs total return ─────────────────────────────────
        ax = axes[1]
        ax.scatter(total_growth_per_sim, w_stars, alpha=0.25, s=8, color=BLUE_MID)
        ax.axhline(0, color=RED, linewidth=1.0, linestyle="--")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_xlabel(f"Total portfolio growth over horizon")
        ax.set_ylabel("Annual withdrawal rate (w*)")
        ax.set_title("Withdrawal Rate vs Total Return")

        plt.tight_layout()
        plt.show()



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
        ax.plot(months, failure_rate, color=RED, linewidth=2)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.tick_params(axis="x", labelrotation=65)
        ax.set_title("Portfolio Depletion Rate", fontweight="bold")
        ax.set_ylabel("Cumulative probability")
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
        inflation        = dict_results.get("sampled_inflation", None)
        if inflation is not None:
            inflation     = inflation[sim_idx, :]
            cum_inflation = np.cumprod(1 + inflation) - 1
        months = range(seq_len)

        # --- Build title ---------------------------------------------------
        title = f"Path Analysis — Rank {rank} of {num_simulations}  |  Final Value: {final_values[sim_idx]:,.0f}"
        if hit_zero[sim_idx]:
            title += f"  |  Depleted at month {first_zero[sim_idx]}"

        # --- Plot ----------------------------------------------------------
        import matplotlib.ticker as mticker
        n_panels = 4 if inflation is not None else 3
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))

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

        # Inflation (monthly bars + cumulative line on twin axis)
        if inflation is not None:
            ax_inf = axes[3]
            ax_inf.bar(months, inflation, color="#f0a855", alpha=0.65, width=1.0, label="Monthly")
            ax_inf.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
            ax_inf.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2%}"))
            ax_inf.set_xlabel("Months drawing down")
            ax_inf.set_ylabel("Monthly Inflation")
            ax_inf.set_title("Inflation")
            ax_inf.grid(alpha=0.25)

            ax_cum = ax_inf.twinx()
            ax_cum.plot(months, cum_inflation, color="#d4720a", linewidth=2, label="Cumulative")
            ax_cum.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
            ax_cum.set_ylabel("Cumulative Inflation")

            lines1, labels1 = ax_inf.get_legend_handles_labels()
            lines2, labels2 = ax_cum.get_legend_handles_labels()
            ax_inf.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc = "upper left")

        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        plt.show()
