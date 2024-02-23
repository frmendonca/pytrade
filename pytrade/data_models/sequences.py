
from pytrade.data_models.options import Option
from pytrade.data_models.stock import Stock

class Sequences():

    """
    Generates random draws to simulate a distribution of returns
    """

    def __init__(
            self,
            base_ticker: str = "SPX",
            base_historical_depth: str = "25y",
            base_returns_freq: int = 30,
            option: Option = None
    ) -> None:
        
        """
        :param base_ticker: the string ticker name of the base asset used to draw returns
        :param base_historical_depth: how far back should we get data for the ticker
        :param return_freq: frequence of returns in sequence
        :param option: an Option object to mix with base_ticker return distribution
        """
        self._base_ticker = base_ticker
        self._base_historical_depth = base_historical_depth
        self._base_returns_freq = base_returns_freq
        self._option = option
        self._specs = None


        def fit(self):
            """
            The method fits the sequences
            """

            # As a first step, load and fetch the data for the base ticker used in the simulation
            # This loads an object Stock that identifies the base ticker historical data
            # We can access this data with self._specs.stock_data
            self._get_initial_specs()
            ...



        def _get_initial_specs(self) -> None:
            """
            Initializes base ticker stock data
            """
            self._specs = Stock(
                ticker = self._base_ticker, 
                historical_depth = self._base_historical_depth,
                returns_freq = self._base_returns_freq
            )

            self._specs.get_stock_data()

        