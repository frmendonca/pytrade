
import argparse
import sys
from pytrade.data_models.portfolio import Portfolio
from pytrade.data_models.sequences import Sequences
from pytrade.data_models.options import Option

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
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Define portfolio
    portfolio = Portfolio(
        shares = dict(zip(args.tickers, args.nb_shares)),
        betas = dict(zip(args.tickers, args.betas))
    )
    portfolio.fit_portfolio()

    # Define option
    if "--options_contracts" in sys.argv:
        option = Option(
            strike = args.strike,
            premium = args.premium,
            days_expiry = args.days_expiry,
            r = args.interest_rate,
            implied_vol = args.implied_vol,
            contracts = args.options_contracts
        )
    


    


