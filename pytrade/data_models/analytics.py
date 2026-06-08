
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pytrade.data_models.simulation import BaseSimulationResults


class PortfolioAnalytics:

    def plot_perpetual_withdrawal_rate_distribution(
        self,
        simulation_results: BaseSimulationResults
    ) -> None:
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
        ax.set_xlabel("Total portfolio growth over horizon")
        ax.set_ylabel("Annual withdrawal rate (w*)")
        ax.set_title("Withdrawal Rate vs Total Return")

        plt.tight_layout()
        plt.show()


    def analyze_sequence_sensitivity(
        self,
        simulation_results: BaseSimulationResults,
    ) -> BaseSimulationResults:
        """
        For each year in the simulation horizon, compute the Spearman rank correlation
        between that year's return and the terminal portfolio value, across all simulated
        paths. Also computes the partial correlation controlling for total path return,
        which isolates the pure timing effect from return magnitude.
        """
        from scipy.stats import spearmanr

        outputs         = simulation_results.simulation_output
        returns_arr     = outputs["sampled_returns"]           # (N, T)
        terminal_values = outputs["portfolio_value"][:, -1]   # (N,)
        _, T            = returns_arr.shape
        horizon_years   = T // 12

        total_return = np.prod(1 + returns_arr, axis=1) - 1
        r_YZ = spearmanr(terminal_values, total_return).statistic

        annual_corr   = np.empty(horizon_years)
        partial_corr  = np.empty(horizon_years)

        for yr in range(horizon_years):
            ann_ret = np.prod(1 + returns_arr[:, yr * 12:(yr + 1) * 12], axis=1) - 1
            r_XY = spearmanr(ann_ret, terminal_values).statistic
            r_XZ = spearmanr(ann_ret, total_return).statistic
            denom = np.sqrt((1 - r_XZ ** 2) * (1 - r_YZ ** 2))
            annual_corr[yr]  = r_XY
            partial_corr[yr] = (r_XY - r_XZ * r_YZ) / denom if denom > 1e-10 else 0.0

        return BaseSimulationResults(simulation_output={
            "years":                      np.arange(1, horizon_years + 1),
            "annual_correlation":         annual_corr,
            "annual_partial_correlation": partial_corr,
        })


    def plot_sequence_sensitivity(
        self,
        sensitivity_results: BaseSimulationResults,
    ) -> None:
        outputs      = sensitivity_results.simulation_output
        years        = outputs["years"]
        raw_corr     = outputs["annual_correlation"]
        partial_corr = outputs["annual_partial_correlation"]

        BLUE_DARK, RED = "#1a4f7a", "#c0392b"

        def bar_colors(values):
            return [BLUE_DARK if v >= 0 else RED for v in values]

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        fig.suptitle("Sequence-of-Returns Sensitivity", fontsize=14, fontweight="bold")

        # ── Left: raw annual Spearman correlation ─────────────────────────
        ax = axes[0]
        ax.bar(years, raw_corr, color=bar_colors(raw_corr), edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Retirement year")
        ax.set_ylabel("Spearman correlation with terminal value")
        ax.set_title("Return Impact by Year")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_xticks(years[::2])

        # ── Right: partial correlation (controlling for total return) ─────
        ax = axes[1]
        ax.bar(years, partial_corr, color=bar_colors(partial_corr), edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Retirement year")
        ax.set_ylabel("Partial correlation with terminal value")
        ax.set_title("Pure Timing Effect by Year\n(controlling for total path return)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_xticks(years[::2])

        plt.tight_layout()
        plt.show()


    def analyze_drawdowns(
        self,
        returns_2d: np.ndarray,
    ) -> BaseSimulationResults:
        """
        Extract all drawdown events from a matrix of bootstrapped return paths and
        pool them into a single distribution. Each row is one simulated path.

        A drawdown event is a contiguous period where the portfolio is below its
        prior peak. Depth is the maximum percentage decline within the event.
        Duration is the number of months from peak to full recovery (or to the end
        of the path if the portfolio never recovers within the horizon).
        """
        def _extract_events(returns_1d):
            wealth = np.cumprod(1 + returns_1d)
            peak   = np.maximum.accumulate(wealth)
            dd     = wealth / peak - 1

            in_dd  = dd < -1e-10
            edges  = np.diff(in_dd.astype(int))
            starts = np.where(edges == 1)[0] + 1
            ends   = np.where(edges == -1)[0] + 1

            if len(in_dd) > 0 and in_dd[-1]:
                ends = np.append(ends, len(dd))

            events = []
            for s, e in zip(starts, ends):
                peak_idx   = s - 1
                trough_idx = s + int(np.argmin(dd[s:e]))
                recovered  = int(e) < len(dd)
                events.append({
                    "depth":          float(dd[trough_idx]),
                    "duration":       int(e) - peak_idx,
                    "time_to_trough": trough_idx - peak_idx,
                    "recovery_time":  (int(e) - trough_idx) if recovered else np.nan,
                    "recovered":      recovered,
                })
            return events

        all_events = []
        for row in returns_2d:
            all_events.extend(_extract_events(row))

        if not all_events:
            empty = np.array([])
            return BaseSimulationResults(simulation_output={
                "depths": empty, "durations": empty,
                "times_to_trough": empty, "recovery_times": empty,
                "recovered": np.array([], dtype=bool),
            })

        return BaseSimulationResults(simulation_output={
            "depths":          np.array([-e["depth"]          for e in all_events]),
            "durations":       np.array([e["duration"]        for e in all_events]),
            "times_to_trough": np.array([e["time_to_trough"]  for e in all_events]),
            "recovery_times":  np.array([e["recovery_time"]   for e in all_events], dtype=float),
            "recovered":       np.array([e["recovered"]       for e in all_events], dtype=bool),
        })


    def plot_drawdown_distribution(
        self,
        drawdown_results: BaseSimulationResults,
    ) -> None:
        outputs   = drawdown_results.simulation_output
        depths    = outputs["depths"]           # ≤ 0
        durations = outputs["durations"]        # months
        recovered = outputs["recovered"]        # bool

        BLUE_MID, RED = "#4a90d9", "#c0392b"
        BLUE_DARK     = "#1a4f7a"

        pct_labels = [50, 95, 99]
        depth_pcts    = np.percentile(depths,    pct_labels)
        duration_pcts = np.percentile(durations, pct_labels)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"Drawdown Distribution  ({len(depths):,} events across {recovered.size} total)",
            fontsize=14, fontweight="bold"
        )

        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
        ax_scatter = fig.add_subplot(gs[:, 0])   # tall left panel
        ax_depth   = fig.add_subplot(gs[0, 1])
        ax_dur     = fig.add_subplot(gs[1, 1])

        # ── Left: scatter depth vs duration ───────────────────────────────
        ax_scatter.scatter(
            durations[recovered],  depths[recovered],
            alpha=0.15, s=6, color=BLUE_MID, label=f"Recovered ({recovered.sum():,})"
        )
        ax_scatter.scatter(
            durations[~recovered], depths[~recovered],
            alpha=0.35, s=10, color=RED, label=f"Unrecovered ({(~recovered).sum():,})"
        )
        ax_scatter.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax_scatter.set_xlabel("Duration (months)")
        ax_scatter.set_ylabel("Depth")
        ax_scatter.set_title("Depth vs Duration")
        ax_scatter.legend(fontsize=8)

        # ── Top-right: depth histogram ─────────────────────────────────────
        ax_depth.hist(depths, bins=60, color=BLUE_MID, edgecolor="white",
                      linewidth=0.3, alpha=0.85)
        pct_colors = ["#e67e22", BLUE_DARK, "#1a6b3c"]
        for pct, val, col in zip(pct_labels, depth_pcts, pct_colors):
            ax_depth.axvline(val, color=col, linewidth=1.4, linestyle="--",
                             label=f"p{pct}: {val:.1%}")
        ax_depth.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax_depth.set_xlabel("Depth")
        ax_depth.set_ylabel("Frequency")
        ax_depth.set_title("Drawdown Depth")
        ax_depth.legend(fontsize=8)

        # ── Bottom-right: duration histogram ──────────────────────────────
        ax_dur.hist(durations, bins=40, color=BLUE_MID, edgecolor="white",
                    linewidth=0.3, alpha=0.85)
        for pct, val, col in zip(pct_labels, duration_pcts, pct_colors):
            ax_dur.axvline(val, color=col, linewidth=1.4, linestyle="--",
                           label=f"p{pct}: {val:.0f}m")
        ax_dur.set_xlabel("Duration (months)")
        ax_dur.set_ylabel("Frequency")
        ax_dur.set_title("Drawdown Duration")
        ax_dur.legend(fontsize=8)

        plt.show()


    def generate_simulation_report(self, simulation_results: BaseSimulationResults):
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

        hit_zero  = (portfolio_arr == 0).any(axis=1)
        first_zero = np.where(
            hit_zero,
            np.argmax(portfolio_arr == 0, axis=1),
            seq_len
        )

        # lexsort: rightmost key = primary sort, leftmost = secondary sort for ties.
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
        n_panels = 4 if inflation is not None else 3
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))

        axes[0].plot(months, portfolio_values, color="darkblue")
        axes[0].set_xlabel("Months drawing down")
        axes[0].set_ylabel("Portfolio Value")
        axes[0].set_title("Portfolio Value")
        axes[0].grid(alpha=0.25)

        axes[1].plot(months, returns, color="darkgreen")
        axes[1].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("Months drawing down")
        axes[1].set_ylabel("Monthly Return")
        axes[1].set_title("Return Sequence")
        axes[1].grid(alpha=0.25)

        axes[2].plot(months, withdrawals, color="darkred")
        axes[2].set_xlabel("Months drawing down")
        axes[2].set_ylabel("Withdrawal Amount")
        axes[2].set_title("Withdrawal Amount")
        axes[2].grid(alpha=0.25)

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
            ax_inf.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        plt.show()
