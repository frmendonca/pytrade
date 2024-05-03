
import numpy as np
import sys
import logging
import pathlib
from pytrade.data_models.portfolio import Portfolio
from pytrade.data_models.sequences import Sequences
from pytrade.data_models.options import Option
from pytrade.simulation import get_args

from pytrade.utils import (
    return_frequency_to_minor,
    compute_compound_returns, 
    compute_statistics, 
    timestamp
)


def main():

    print("Simulation initiated...")

    # Define portfolio
    portfolio = Portfolio(
        shares = dict(zip(args.tickers, args.nb_shares)),
        betas = dict(zip(args.tickers, args.betas))
    )
    portfolio.fit_portfolio()

    logger.info("\nPortfolio")
    logger.info(portfolio.shares)
    logger.info(f"Market value {portfolio.market_value}")

    # Define option
    if "-oc" in sys.argv:
        option = Option(
            strike = args.strike,
            premium = args.premium,
            days_expiry = args.days_expiry,
            r = args.interest_rate,
            iv = args.implied_vol,
            contracts = args.options_contracts
        )

        logger.info("\nOption definition")
        logger.info(f"Strike:{option.strike}; Premium:{option.premium}; DTE:{option.days_expiry}; IV:{option.iv}; Contracts:{option.contracts}")

    else:
        option = Option()
        logger.info(f"Strike:{np.nan}; Premium:{np.nan}; DTE:{np.nan}; IV:{np.nan}; Contracts:{0}")

    # Define sequences
    sequences = Sequences(
        base_historical_depth=args.historical_depth,
        base_returns_freq = args.returns_freq,
        portfolio=portfolio,
        option = option
    )
    sequences.fit() # Fit sequences

    number_of_periods = args.sim_number_years*return_frequency_to_minor(args.returns_freq)
    return_sequences = np.array(
        [
            np.random.choice(sequences.sequence, size = number_of_periods, replace = True)
            for _ in range(200000)
        ]
    )
    compound_returns = compute_compound_returns(return_sequences)
    statistics = compute_statistics(compound_returns, args.sim_number_years)
    logger.info("\nSimulation results")
    logger.info(statistics)

    print("Simulation completed!")
    
if __name__ == "__main__":

    logger = logging.getLogger("simulation_results")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{pathlib.Path(__file__).parent.absolute()}/{timestamp()}.log")
    logger.addHandler(fh)

    args = get_args()
    main()
    