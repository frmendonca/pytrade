
import math

from dataclasses import dataclass
from scipy.stats import norm


@dataclass(repr = True)
class Option:
    """
    Defines an option contract
    """
    strike: float
    premium: float
    days_expiry: int
    r: float
    implied_vol: float
    contracts: int = 0
    direction: str = "P"


    def intrinsic_value(self, underlying: float) -> float:
        """
        Computes an option intrinsic value
        """
        if self.direction == "P":
            return self.contracts*max(self.strike - underlying, 0)
        else:
            return self.contracts*max(underlying - self.strike, 0)
        
    
    def compute_black_scholes_price(self, underlying) -> float:
        """
        Compute an option price using the Black-Scholes formula
        """
        if self.direction == "P":
            d1 = (math.log(underlying / self.strike) + (self.r + 0.5 * self.implied_vol ** 2) * self.days_expiry) / (self.implied_vol * math.sqrt(self.days_expiry))
            d2 = d1 - self.implied_vol * math.sqrt(self.days_expiry)

            put_price = self.strike * math.exp(-self.r * self.days_expiry) * norm.cdf(-d2) - underlying * norm.cdf(-d1)

            return put_price