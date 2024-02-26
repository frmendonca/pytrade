
import argparse
import numpy as np
import sys
import logging
import pathlib
from pytrade.data_models.portfolio import Portfolio
from pytrade.data_models.sequences import Sequences
from pytrade.data_models.options import Option

from pytrade.utils import (
    return_frequency_to_minor,
    compute_compound_returns, 
    compute_statistics, 
    timestamp
)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t","--tickers",
        required=True,
        nargs="+",
        help="Tickers composing the portfolio"
    ),
    parser.add_argument(
        "-q","--nb_shares",
        required=True,
        type=int,
        nargs="+",
        help="Number of shares for each ticker. Should follow the same order as -p arg"
    ),
    parser.add_argument(
        "-bt","--betas",
        required=True,
        type=float,
        nargs="+",
        help="Betas associated with each ticker. Should follow the same order as -p arg"
    ),
    parser.add_argument(
        "-oc","--options_contracts",
        required=False,
        type = int,
        help="Number of options contracts considered in the simulation"
    ),
    parser.add_argument(
        "-k","--strike",
        required="-oc" in sys.argv,
        type=float,
        help="Option strike price"
    ),
    parser.add_argument(
        "-dte","--days_expiry",
        required="-oc" in sys.argv,
        type=int,
        help="Option days to expire"
    ),
    parser.add_argument(
        "-ri","--interest_rate",
        required="-oc" in sys.argv,
        type=float,
        help="Option implied risk free rate"
    ),
    parser.add_argument(
        "-iv","--implied_vol",
        required="-oc" in sys.argv,
        type=float,
        help="Option implied volatility"
    ),
    parser.add_argument(
        "-p","--premium",
        required="-oc" in sys.argv,
        type=float,
        help="Option premium"
    ),
    parser.add_argument(
        "-hd","--historical_depth",
        required=False,
        type=str,
        help="Historical depth of option base ticker, in years."
    ),
    parser.add_argument(
        "-sf","--returns_freq",
        required=False,
        type=int,
        help="Time frequency of option base ticker, in days. Example: -sfd = 30 means we simulate with monthly returns"
    ),
    parser.add_argument(
        "-sny","--sim_number_years",
        required=False,
        type=int,
        help="Number of years to run the simulation"
    )
    return parser.parse_args()


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
            for _ in range(50000)
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

    
    


    


