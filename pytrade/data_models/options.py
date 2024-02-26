
import math
import numpy as np

from dataclasses import dataclass
from scipy.stats import norm


@dataclass(repr = True)
class Option:
    """
    Defines an option contract
    """
    strike: float = None
    premium: float = None
    days_expiry: int = None
    r: float = None
    iv: float = None
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
        Compute the Black-Scholes price of a put option.
        :param underlying - underlying market price
        :returns float: Black-Scholes put option price
        """
        d1 = (np.log(underlying / self.strike) + (self.r + 0.5 * self.iv ** 2) * self.days_expiry/365) / (self.iv * np.sqrt(self.days_expiry/365))
        d2 = d1 - self.iv * np.sqrt(self.days_expiry/365)
        put_price = self.strike * np.exp(-self.r * self.days_expiry/365) * norm.cdf(-d2) - underlying * norm.cdf(-d1)
        return put_price
