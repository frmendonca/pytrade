
import argparse
import sys

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