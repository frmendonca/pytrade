
import numpy as np
from typing import Final
from scipy.stats import norm
from datetime import datetime
from enum import StrEnum
from dataclasses import dataclass

RISK_FREE_RATE: Final[float] = 0.035


class OptionType(StrEnum):
    PUT = "PUT"
    CALL = "CALL"


class OptionDirection(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Greeks:
    delta: float
    theta: float
    gamma: float
    vega: float


class OptionModel:
    def __init__(
        self,
        ticker: str,
        strike: float,
        premium: float,
        iv: float,
        expiration_date: str,
        option_type: OptionType,
        option_direction: OptionDirection
    ):

        self.ticker = ticker
        self.strike = strike
        self.premium = premium
        self.iv = iv
        self.expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        self.option_type = option_type
        self.option_direction = option_direction

        # Freeze DTE at construction time so simulation results are reproducible
        # regardless of when the notebook cell is re-run within a session.
        self._days_to_expiry: int = (self.expiration_date - datetime.today()).days


    @property
    def direction_multiplier(self) -> int:
        return -1 if self.option_direction == OptionDirection.SHORT else 1


    @property
    def days_to_expiry(self) -> int:
        """DTE as captured at construction time (wall-clock-independent)."""
        return self._days_to_expiry

    @property
    def live_days_to_expiry(self) -> int:
        """Real-time DTE based on today's date. Use outside of simulation paths."""
        return (self.expiration_date - datetime.today()).days


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
        if years_to_expiry <= 0.0001 / 365.0:
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
    

    def compute_greeks(
        self,
        underlying_price: float,
        iv: float | None = None,
        days_to_expiry: float | None = None
    ) -> Greeks:
        
        if iv is None:
            iv = self.iv
        if days_to_expiry is None:
            days_to_expiry = self.days_to_expiry

        years_to_expiry = max(days_to_expiry, 0.0001) / 365.0
        if years_to_expiry <= 0.0001 / 365.0:
            if self.option_type == OptionType.CALL:
                delta = 1.0 if underlying_price > self.strike else 0.0
            else:
                delta = -1.0 if underlying_price < self.strike else 0.0
            return Greeks(delta=delta, theta=0.0, gamma=0.0, vega=0.0)


        d1, d2 = self._compute_d1_d2(underlying_price, self.strike, years_to_expiry, iv)

        pdf_d1 = norm.pdf(d1)
        sqrt_t = np.sqrt(years_to_expiry)

        # Gamma (same for calls and puts)
        gamma = pdf_d1 / (underlying_price * iv * sqrt_t)

        # Vega (same for calls and puts, per 1% move in IV)
        vega = underlying_price * pdf_d1 * sqrt_t * 0.01

        if self.option_type == OptionType.CALL:
            delta = norm.cdf(d1)
            theta = (
                -(underlying_price * pdf_d1 * iv) / (2 * sqrt_t)
                - RISK_FREE_RATE * self.strike * np.exp(-RISK_FREE_RATE * years_to_expiry) * norm.cdf(d2)
            ) / 365.0

        else:
            delta = norm.cdf(d1) - 1
            theta = (
                -(underlying_price * pdf_d1 * iv) / (2 * sqrt_t)
                + RISK_FREE_RATE * self.strike * np.exp(-RISK_FREE_RATE * years_to_expiry) * norm.cdf(-d2)
            ) / 365.0

        return Greeks(delta=delta, theta=theta, gamma=gamma, vega=vega)

       

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
                

@dataclass
class OptionLeg:
    option: OptionModel
    ncontracts: int


class OptionStrategy:
    def __init__(self, legs: list[OptionLeg]):
        self.legs = legs


    @property
    def strategy_premium(self) -> float:
        total_premium = 0
        for leg in self.legs:
            total_premium -= leg.option.direction_multiplier * leg.option.premium * leg.ncontracts
        return total_premium


    def strategy_greeks(
        self,
        underlying_price: float,
        iv_factor: float | None = None,
        days_elapsed: float = 0.0
    ) -> Greeks:
        if iv_factor is None:
            iv_factor = 1.0

        total = Greeks(delta=0.0, theta=0.0, gamma=0.0, vega=0.0)

        for leg in self.legs:
            opt = leg.option
            g = opt.compute_greeks(
                underlying_price,
                iv=opt.iv * iv_factor,
                days_to_expiry=opt.days_to_expiry - days_elapsed
            )
            n = leg.ncontracts * opt.direction_multiplier
            total.delta += g.delta * n
            total.theta += g.theta * n
            total.gamma += g.gamma * n
            total.vega  += g.vega  * n

        return total
    

    def resolve_value(
        self,
        underlying_price: float,
        iv_factor: float | None = 1.0,
        days_elapsed: float = 0
    ):
        
        if iv_factor is None:
            iv_factor = 1.0
        
        strategy_value = 0
        for leg in self.legs:
            opt = leg.option
            opt_price = (
                opt.direction_multiplier 
                * opt.black_scholes_calculation(underlying_price, opt.iv * iv_factor, opt.days_to_expiry - days_elapsed)
                * leg.ncontracts
            )
            strategy_value += opt_price

        return strategy_value

