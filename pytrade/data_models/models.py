
import numpy as np
from typing import Final
from scipy.stats import norm
from enum import StrEnum

RISK_FREE_RATE: Final[float] = 0.035

class OptionType(StrEnum):
    PUT = "PUT"
    CALL = "CALL"


class OptionDirection(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class OptionModel:
    def __init__(
        self,
        ticker: str,
        strike: float,
        premium: float,
        days_to_expiry: int,
        option_type: str,
        option_direction: str
    ):

        self.ticker = ticker
        self.strike = strike
        self.premium = premium
        self.days_to_expiry = days_to_expiry
        self.option_type = option_type
        self.option_direction = option_direction

        self._option_direction_sign = (-1)**(self.option_direction == OptionDirection.SHORT)


    def black_scholes_calculation(
        self,
        underlying_price: float,
        iv: float,
        days_to_expiry: float | None = None
    ):
        if days_to_expiry is None:
            days_to_expiry = self.days_to_expiry

        # Convert days to years
        T = days_to_expiry / 365.

        match self.option_type:
            case OptionType.CALL:
                return self._black_scholes_call(
                            underlying_price,
                            self.strike,
                            T,
                            iv
                        )

            case OptionType.PUT:
                return self._black_scholes_put(
                            underlying_price,
                            self.strike,
                            T,
                            iv
                        )

            case _:
                NotImplementedError("Option type has to be Call or Put")

    @staticmethod
    def _black_scholes_put(
        underlying_price: float,
        strike: float,
        days_to_expiry: int,
        iv: float
    ):
        """
        underlying_price: Current stock price
        strike: Strike price
        days_to_expiry: Time to maturity (in years)
        interest_rate: Risk-free interest rate (decimal)
        iv: Volatility of the underlying asset (decimal)
        """

        # Calculate d1 and d2
        d1 = ((np.log(underlying_price / strike) + (RISK_FREE_RATE + 0.5 * iv ** 2) * days_to_expiry) /
              (iv * np.sqrt(days_to_expiry)))
        d2 = d1 - iv * np.sqrt(days_to_expiry)

        # Calculate Put price
        put_price = strike * np.exp(-RISK_FREE_RATE * days_to_expiry) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)

        return put_price


    @staticmethod
    def _black_scholes_call(
        underlying_price: float,
        strike: float,
        days_to_expiry: int,
        iv: float
    ):
        """
        underlying_price: Current stock price
        strike: Strike price
        days_to_expiry: Time to maturity (in years)
        interest_rate: Risk-free interest rate (decimal)
        iv: Volatility of the underlying asset (decimal)
        """

        # Calculate d1 and d2
        d1 = ((np.log(underlying_price / strike) + (RISK_FREE_RATE + 0.5 * iv ** 2) * days_to_expiry) /
              (iv * np.sqrt(days_to_expiry)))
        d2 = d1 - iv * np.sqrt(days_to_expiry)

        # Calculate Put price
        call_price = underlying_price * norm.cdf(d1) - strike * np.exp(-RISK_FREE_RATE * days_to_expiry) * norm.cdf(d2)

        return call_price


