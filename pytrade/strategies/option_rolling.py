import argparse
from pathlib import Path
from pytrade.utils import read_json


DATA_PATH = Path(__file__).parent.absolute() / "data"


def calculate_roll_details(
    current_strike: float,
    current_premium_received: float,
    number_contracts_sold: float,
    current_premium_to_close: float,
    candidate_strike: float,
    candidate_premium: float,
    underlying_price: float,
) -> dict[str, float]:

    multiple = number_contracts_sold * 100
    current_pnl = (current_premium_received - current_premium_to_close) * multiple

    post_roll_net_credit_received = (
        candidate_premium + current_premium_received - current_premium_to_close
    ) * multiple
    rolled_put_value_to_breakeven = candidate_premium + (
        current_premium_received - current_premium_to_close
    )

    # Margin computations
    pre_otm_amount = max(underlying_price - current_strike, 0)
    pre_roll_margin = (
        max(0.20 * underlying_price - pre_otm_amount, 0.10 * current_strike)
        + current_premium_received
    )
    pre_roll_margin *= multiple

    post_otm_amount = max(underlying_price - candidate_strike, 0)
    post_roll_margin = (
        max(0.20 * underlying_price - post_otm_amount, 0.10 * candidate_strike)
        + candidate_premium
    )
    post_roll_margin *= multiple

    post_roll_margin_change = pre_roll_margin - post_roll_margin

    return {
        "current_pnl": current_pnl,
        "post_roll_net_credit": post_roll_net_credit_received,
        "post_roll_premium_breakeven": rolled_put_value_to_breakeven,
        "pre_roll_margin": pre_roll_margin,
        "post_roll_margin": post_roll_margin,
        "post_roll_margin_change": post_roll_margin_change,
    }


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Calculate P&L for rolling put options."
    )
    parser.add_argument("--current_strike", type=float, help="Current put strike price")
    parser.add_argument(
        "--current_premium_received",
        type=float,
        help="Premium received for current puts",
    )
    parser.add_argument("--number_contracts_sold", type=int, help="Number of contracts")
    parser.add_argument(
        "--current_premium_to_close", type=float, help="Current premium to close puts"
    )
    parser.add_argument("--underlying_price", type=float, help="Underlying asset price")

    # Parse arguments
    args = parser.parse_args()

    # Assign parsed arguments to variables
    current_strike = args.current_strike
    current_premium_received = args.current_premium_received
    number_contracts_sold = args.number_contracts_sold
    current_premium_to_close = args.current_premium_to_close
    underlying_price = args.underlying_price

    # Read data
    data = read_json(DATA_PATH / "option_chain.json")
    option_chain = data["puts"]

    # Loop through each put option and compute details
    print(f"Rolling Options (NVDA at ${underlying_price}):")
    for opt in option_chain:
        candidate_strike = opt["strike"]
        candidate_premium = opt["premium"]
        result = calculate_roll_details(
            current_strike,
            current_premium_received,
            number_contracts_sold,
            current_premium_to_close,
            candidate_strike,
            candidate_premium,
            underlying_price,
        )

        print(candidate_strike, {k: round(v, 2) for k, v in result.items()})


if __name__ == "__main__":
    main()
