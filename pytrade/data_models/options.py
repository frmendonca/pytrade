
import math

from dataclasses import dataclass
from scipy.stats import norm


@dataclass(repr = True)
class Option:
    """
    Defines an option contract
    """
    direction: str = "P"
    strike: float
    premium: float
    contracts: int = 1


    def intrinsic_value(self, underlying: float) -> float:
        """
        Computes an option intrinsic value
        """
        if self.direction == "P":
            return self.contracts*max(self.strike - underlying, 0)
        else:
            return self.contracts*max(underlying - self.strike, 0)
        
    
    def compute_black_scholes_price(self, underlying, t, r, s) -> float:
        """
        Compute an option price using the Black-Scholes formula
        """

        d1 = (math.log(underlying / self.strike) + (r + 0.5 * s ** 2) * t) / (s * math.sqrt(t))
        d2 = d1 - s * math.sqrt(t)

        put_price = self.strike * math.exp(-r * t) * norm.cdf(-d2) - underlying * norm.cdf(-d1)

        return put_price