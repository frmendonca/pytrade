
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from pytrade.data_models.simulation import BaseSimulationModel, BaseSimulationResults
from pytrade.data_models.analytics import PortfolioAnalytics


@dataclass
class StockPosition:
    ticker: str
    allocation: float = 1.0
    leverage: float = 1.0
    expense_ratio: float = 0.001



class Portfolio(BaseSimulationModel, PortfolioAnalytics):

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
