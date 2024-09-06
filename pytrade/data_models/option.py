import numpy as np
from scipy.stats import norm
from datetime import datetime
from scipy.optimize import minimize
from pytrade.data_models.stock import Stock


class OptionUnderlying:
    def __init__(self, ticker: str, **kwargs):
        self.ticker = ticker

        # Obtain underlying data
        self.stock = Stock(ticker = ticker, quantity = 100, beta = 1.0)
        self.stock.load_stock_data(freq=kwargs["kwargs"].get("freq", 1))


class Option:
    """
    The Option object.
    Represents a put option

    :ivar expiry: date the date of option expiry
    :ivar strike: float the strike price
    :ivar premium: float the option's premium
    """
    def __init__(
        self,
        strike: float,
        premium: float,
        expiry: str,
        contracts: int,
        underlying: OptionUnderlying,
        **kwargs
    ):
        self.strike=strike
        self.expiry=datetime.strptime(expiry, "%Y-%m-%d").date()
        self.premium=premium
        self.contracts=contracts
        self.underlying=underlying
        
        # Compute DTE
        self.days_to_expiry=(self.expiry - datetime.now().date()).days

        # Fetch risk free rate
        self.interest_rate=kwargs["kwargs"].get("risk_free_rate", 0.05) # TODO: Should probably have something for this

        # Calibrate iv
        self.calibrate_iv()


    def __repr__(self):
        cls=self.__class__.__name__
        return f"{cls}(ticker={self.underlying.ticker}, strike={self.strike}, premium={self.premium}, contracts={self.contracts}, dte={self.days_to_expiry})"  


    def compute_black_scholes_put_option_price(self, underlying_price, days_to_expiry) -> float:
        '''
        Compute the price of a put option using the Black Scholes
        model
        '''
        d1 = (np.log(underlying_price / self.strike) + (self.interest_rate + 0.5 * self.iv ** 2) * days_to_expiry/365) / (self.iv * np.sqrt(days_to_expiry/365))
        d2 = d1 - self.iv * np.sqrt(days_to_expiry/365)
        put_price = self.strike * np.exp(-self.interest_rate * days_to_expiry/365) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)
        return put_price.item()
    

    def option_intrinsic_value(self, underlying_price: float) -> float:
        """
        Computes an option's intrinsic value
        :param underlying: float the underlying asset's current price
        :return float the option's intrinsic value at expiration
        """
        return max(self.strike - underlying_price, 0)


    def calibrate_iv(self):
        '''
        Given the option premium and current underlying price,
        computes the IV
        '''
        self.iv = 0.17 # Starting value

        def fn_to_minimize(iv):
            self.iv = iv
            model_price = self.compute_black_scholes_put_option_price(
                underlying_price=self.underlying.stock.ticker_price,
                days_to_expiry=self.days_to_expiry
            )
            true_price = self.premium
            return (model_price - true_price)**2


        optim = minimize(fun=fn_to_minimize, x0=0.15)
        self.iv = optim.x




