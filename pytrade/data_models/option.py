
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
    :ivar contracts: int the number of contracts, where each unit represents 100 stocks
    """
    def __init__(
        self,
        strike: int,
        premium: float,
        iv: float,
        r: float,
        option_type: OptionType,
        option_direction: OptionDirection,
        expiration_date: str,
        contracts: int
    ):
        self._strike = strike
        self._premium = premium
        self._iv = iv
        self._r = r
        self._option_type = option_type
        self._option_direction = option_direction
        self._expiration_date = expiration_date
        self._contracts = contracts
        self._days_to_expiration = (
            datetime.strptime(expiration_date, "%Y-%m-%d") - datetime.now()
        ).days


    def __repr__(self):
        cls=self.__class__.__name__
        return f"{cls}(strike={self._strike}, premium={self._premium}, dte={self._days_to_expiration}, iv={self._iv}, r={self._r})"  


    def compute_intrinsic_value(self, underlying: float):
        if self._option_type == OptionType.CALL:
            return 100*self._contracts*max(underlying - self._strike, 0)
        else:
            return 100*self._contracts*max(self._strike - underlying, 0)
    

    def compute_black_scholes_value(
        self,
        underlying: float,
        iv: float | None = None,
        days_to_expiry: int | None = None
    ):
        """
        Computes the value of the option using the black scholes 
        formula.

        :underlying a float value representing the current underlying value
        :iv the implied volatility used to compute the formula
        :r the risk free interest rate
        :days_to_expiry optional parameter to compute the value at a specific DTE. 
            If None it takes the option original DTE
        """
        
        if iv is None:
            iv = self._iv

        if days_to_expiry is None:
            days_to_expiry = self._days_to_expiration

        T = days_to_expiry / 365.0

        # Calculate d1 and d2
        d1 = (math.log(underlying / self._strike) + (self._r + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)

        if self._option_type == OptionType.CALL:
            price = underlying * norm.cdf(d1) - self._strike * math.exp(-self._r * T) * norm.cdf(d2)
            return price
        
        elif self._option_type == OptionType.PUT:
            price = self._strike * math.exp(-self._r * T) * norm.cdf(-d2) - underlying * norm.cdf(-d1)
            return price

