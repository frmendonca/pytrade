
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
        iv: float,
        days_to_expiry: int,
        option_type: OptionType,
        option_direction: OptionDirection
    ):

        self.ticker = ticker
        self.strike = strike
        self.premium = premium
        self.iv = iv
        self.days_to_expiry = days_to_expiry
        self.option_type = option_type
        self.option_direction = option_direction


    @property
    def direction_multiplier(self) -> int:
        return -1 if self.option_direction == OptionDirection.SHORT else 1


    def black_scholes_calculation(
        self,
        underlying_price: float,
        iv: float | None = None,
        days_to_expiry: float | None = None
    ):
        
        """
        underlying_price: Current stock price
        strike: Strike price
        days_to_expiry: Time to maturity (in days)
        iv: Volatility of the underlying asset (decimal)
        """

        if iv is None:
            iv = self.iv

        if days_to_expiry is None:
            days_to_expiry = self.days_to_expiry

        # Convert days to years
        years_to_expiry = max(days_to_expiry, 0.0001) / 365.
    
        # Exit if almost at maturity
        if years_to_expiry <= 0.0001 / 365.:
            if self.option_type == OptionType.PUT:
                return max(self.strike - underlying_price, 0)
            else:
                return max(underlying_price - self.strike, 0)
            

        # Calculate d1 and d2
        d1, d2 = self._compute_d1_d2(underlying_price, self.strike, years_to_expiry, iv)
        match self.option_type:
            case OptionType.PUT:
                # Calculate Put price
                opt_price = self.strike * np.exp(-RISK_FREE_RATE * years_to_expiry) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)
                            
            case OptionType.CALL:
                # Calculate Call price
                opt_price = underlying_price * norm.cdf(d1) - self.strike * np.exp(-RISK_FREE_RATE * years_to_expiry) * norm.cdf(d2)

            case _:
                raise RuntimeError("Options are either Put or Call")


        return opt_price

       

    @staticmethod
    def _compute_d1_d2(
        underlying_price: float,
        strike: float,
        years_to_expiry: float,
        iv: float
    ):
        
        # Calculate d1 and d2
        d1 = ((np.log(underlying_price / strike) + (RISK_FREE_RATE + 0.5 * iv ** 2) * years_to_expiry) /
              (iv * np.sqrt(years_to_expiry)))
        d2 = d1 - iv * np.sqrt(years_to_expiry)

        return d1, d2
                



class OptionStrategy:
    def __init__(self, options: list[OptionModel], ncontracts: list[int]):
        self.options = options
        self.ncontracts = ncontracts


    @property
    def strategy_premium(self) -> float:
        total_premium = 0
        for i, opt in enumerate(self.options):
            total_premium -= opt.direction_multiplier * opt.premium * self.ncontracts[i]

        return total_premium


    def resolve_value(
        self,
        underlying_price: float,
        iv_factor: float | None,
        days_elapsed: float
    ):
        
        if iv_factor is None:
            iv_factor = 1.0
        
        strategy_value = 0
        for i, opt in enumerate(self.options):
            opt_price = (
                opt.direction_multiplier 
                * opt.black_scholes_calculation(underlying_price, opt.iv * iv_factor, opt.days_to_expiry - days_elapsed)
                * self.ncontracts[i]
            )
            strategy_value += opt_price

        return strategy_value
