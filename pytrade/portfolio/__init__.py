
import pathlib
from pytrade.data_models.stock import Stock
from pytrade.data_models.base import Currency
from pytrade.utils.utils import read_json


CONFIG_PATH = pathlib.Path(__file__).parent.absolute() / "configs/portfolio_config.json"
config = read_json(CONFIG_PATH)

PORTFOLIO = []
for ticker, ticker_info in config.items():
    PORTFOLIO.append(
        Stock(
            ticker=ticker,
            quantity=ticker_info["quantity"],
            cost_basis=ticker_info["cost_basis"],
            dividend=ticker_info["dividends"],
            currency=Currency[ticker_info["currency"]]
        )
    )