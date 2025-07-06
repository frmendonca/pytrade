import math

from enum import Enum
from scipy.stats import norm
from datetime import datetime


class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"


class OptionDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Option:
    """
    An option object

    :ivar strike: int the strike price of the option
    :ivar premium: float the premium of the option
    :ivar option_type: OptionType sets the type of the option as CALL or PUT
    :ivar option_direction: OptionDirection sets the option as LONG or SHORT
    :ivar expiration_date: str expiration date in format yyyy-mm-dd
    :ivar quantity: int the number of contracts, where each unit represents 100 stocks
    """

    def __init__(
        self,
        ticker: str,
        strike: int,
        premium: float,
        iv: float,
        r: float,
        option_type: OptionType,
        option_direction: OptionDirection,
        expiration_date: str,
        quantity: int,
    ):
        self.ticker = ticker
        self.strike = strike
        self.premium = premium
        self.iv = iv
        self.r = r
        self.option_type = option_type
        self.option_direction = option_direction
        self.expiration_date = expiration_date
        self.quantity = quantity

        try:
            self.days_to_expiration = (
                datetime.strptime(expiration_date, "%Y-%m-%d") - datetime.now()
            ).days
        except ValueError:
            raise ValueError(f"Invalid expiration date format: '{expiration_date}'. Expected YYYY-mm-dd.")

        # Call the validation method after all attributes are set
        self._validate_instance_variables()

    def _validate_instance_variables(self):
        """
        Validates the instance variable for the Option object
        Raises ValueError if any validation fails
        """

        if not isinstance(self.strike, (int, float)) or self.strike <= 0:
            raise ValueError(f"Strike price must be a positive number. Got {self.strike}")
        if not isinstance(self.premium, (int, float)) or self.premium <= 0:
            raise ValueError(f"Premium must be a positive number. Got {self.premium}")
        if not isinstance(self.iv, (int, float)) or self.iv <= 0:
            raise ValueError(f"IV must be a positive number. Got {self.iv}")
        if not isinstance(self.r, (int, float)) or self.r < 0:
            raise ValueError(f"Interest rate must be a positive number. Got {self.r}")
        if not isinstance(self.quantity, (int, float)) or self.quantity <= 0:
            raise ValueError(f"Number of contracts must be a positive number. Got {self.quantity}")


    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(strike={self.strike}, premium={self.premium}, dte={self.days_to_expiration}, iv={self.iv}, r={self.r})"


    def compute_intrinsic_value(self, underlying: float):
        if self.option_type == OptionType.CALL:
            return self.quantity * max(underlying - self.strike, 0)
        else:
            return self.quantity * max(self.strike - underlying, 0)


    def compute_black_scholes_value(
        self,
        underlying: float,
        iv: float | None = None,
        days_to_expiry: int | None = None,
    ) -> float:
        """
        Computes the value of the option using the black scholes
        formula.

        :param underlying a float value representing the current underlying value
        :param iv the implied volatility used to compute the formula
        :param r the risk free interest rate
        :param days_to_expiry optional parameter to compute the value at a specific DTE.
            If None it takes the option original DTE
        :returns the value of the option
        """

        if iv is None:
            iv = self.iv

        if days_to_expiry is None:
            days_to_expiry = self.days_to_expiration

        years_to_expiry = days_to_expiry / 365.0

        # Calculate d1 and d2
        d1 = (math.log(underlying / self.strike) + (self.r + 0.5 * iv**2) * years_to_expiry) / (
            iv * math.sqrt(years_to_expiry)
        )
        d2 = d1 - iv * math.sqrt(years_to_expiry)

        if self.option_type == OptionType.CALL:
            price = underlying * norm.cdf(d1) - self.strike * math.exp(
                -self.r * years_to_expiry
            ) * norm.cdf(d2)


        elif self.option_type == OptionType.PUT:
            price = self.strike * math.exp(-self.r * years_to_expiry) * norm.cdf(
                -d2
            ) - underlying * norm.cdf(-d1)

        else:
            price = None

        return price


    def compute_greeks(
        self,
        underlying: float,
        iv: float | None = None,
        days_to_expiry: int | None = None,
    ) -> dict[str, float]:
        """
        Computes options greeks based on Black Scholes formula
        :param underlying a float value representing the current underlying value
        :param iv the implied volatility used to compute the formula
        :param days_to_expiry optional parameter to compute the value at a specific DTE.
            If None it takes the option original DTE
        :returns the greeks
        """

        if iv is None:
            iv = self.iv

        if days_to_expiry is None:
            days_to_expiry = self.days_to_expiration

        T = days_to_expiry / 365.0

        d1 = 1/(iv*math.sqrt(T))*(math.log(underlying/self.strike) + (self.r + 0.5*iv**2)*T)
        d2 = d1 - iv*math.sqrt(T)

        return {
            "delta": norm.cdf(d1) if self.option_type == OptionType.CALL else norm.cdf(d1) - 1,
            "gamma": norm.pdf(d1) / (underlying * iv * math.sqrt(T)),
            "vega": underlying * norm.pdf(d1) * math.sqrt(T),
            "theta": (
                -underlying * norm.pdf(d1) * iv / (2 * math.sqrt(T)) - self.r * self.strike * math.exp(-self.r * T) * norm.cdf(d2)
                if self.option_type == OptionType.CALL
                else -underlying * norm.pdf(d1) * iv / (2 * math.sqrt(T)) + self.r * self.strike * math.exp(-self.r * T) * norm.cdf(d2)
            )
        }
