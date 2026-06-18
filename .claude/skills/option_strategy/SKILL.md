# Option Strategy Simulator Skill

When invoked, build and execute an option strategy simulation using pytrade's `OptionStrategySimulator` and print the performance report.

---

## Input parsing

Collect the following from the user's message. Ask only for missing required fields.

**Required per leg:**
| Field | Description |
|---|---|
| `ticker` | Underlying ticker (e.g. SPY) — shared across all legs |
| `option_type` | `PUT` or `CALL` |
| `direction` | `SHORT` or `LONG` |
| `strike` | Strike price |
| `premium` | Option premium paid/received |
| `iv` | Implied volatility as decimal (e.g. `0.15` for 15%) |
| `expiration_date` | `YYYY-MM-DD` |
| `ncontracts` | Number of contracts |

**Required global:**
| Field | Default | Description |
|---|---|---|
| `starting_underlying` | required | Current spot price of the underlying |
| `target_ann_vol` | average IV across all legs | Vol used to rescale bootstrap paths |
| `vol_skew` | `2.0` for equity indices, `0.0` for FX/commodities | Spot/vol correlation coefficient |
| `num_resamples` | `20000` | Number of simulation paths |
| `block_length` | `max(5, DTE // 3)` | Mean block length for stationary bootstrap |
| `seed` | `42` | RNG seed for reproducibility |

---

## Steps

1. **Summarise inputs** — print a brief confirmation of the parsed legs and simulation parameters before running (one-line per leg: direction, type, strike, premium, IV, expiry, contracts).
2. **Generate and run the script** — pipe the script directly to `poetry run python` via a Bash heredoc (no temp file needed). Run from the pytrade project root: `cd C:/Users/franc/Documents/pytrade && poetry run python - <<'EOF' ... EOF`
3. **Show results** — display the printed performance table as-is. Follow with a 2–3 sentence interpretation highlighting: Probability of Profit at expiration, Expected Profit, and VaR (1%).

---

## Script template

Substitute all `<placeholders>` with the parsed values. For `target_ann_vol`, compute the average IV of all legs if the user did not specify one.

Use the Bash tool with this exact structure (the `'EOF'` single-quoted delimiter prevents the shell from expanding anything inside):

```bash
cd C:/Users/franc/Documents/pytrade && poetry run python - <<'EOF'
from pytrade.data_models.options import (
    OptionModel, OptionDirection, OptionType, OptionStrategy, OptionLeg
)
from pytrade.simulation.option_strategy import OptionStrategySimulator

strategy = OptionStrategy(
    legs=[
        # Repeat OptionLeg(...) for every leg the user provided
        OptionLeg(
            option=OptionModel(
                ticker="<ticker>",
                strike=<strike>,
                premium=<premium>,
                iv=<iv>,
                expiration_date="<expiration_date>",
                option_type=OptionType.<PUT|CALL>,
                option_direction=OptionDirection.<SHORT|LONG>,
            ),
            ncontracts=<ncontracts>,
        ),
    ]
)

sim = OptionStrategySimulator(strategy=strategy)
blocks = sim.generate_bootstrap_blocks(
    num_resamples=<num_resamples>,
    block_length=<block_length>,
    seed=<seed>,
    target_ann_vol=None,
)
result = sim.simulate_pnl(
    blocks,
    starting_underlying=<starting_underlying>,
    vol_skew=<vol_skew>,
    n_cores=1,
)
OptionStrategySimulator.report_strategy_performance(result, plot=False)
EOF
```

---

## Notes

- `plot=False` must always be passed — plots cannot render in the terminal.
- If `expiration_date` is already in the past when the script runs, `days_to_expiry` will be 0 or negative and Black-Scholes will treat the option as expired. Warn the user if DTE ≤ 0.
- The simulator fetches historical returns from Yahoo Finance via `yfinance`; a network connection is required.
- Multi-leg strategies must all share the same underlying ticker.
- The dolar values in the output report should be multiplied by 100 to be interpreted

