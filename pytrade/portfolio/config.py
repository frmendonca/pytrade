
from pytrade.data_models.stock import Stock
from pytrade.data_models.base import Currency

PORTFOLIO = [
    Stock(ticker="NVO", quantity=126, avg_cost=72.24, dividend = 1.14*4, currency=Currency.USD),
    Stock(ticker="VWCE.DE", quantity=96, avg_cost=128.93, currency=Currency.USD),
    Stock(ticker="SXRV.DE", quantity=5, avg_cost=0.0, currency=Currency.USD),
    Stock(ticker="SCHD", quantity=400, avg_cost=27.00, dividend=0.26*4, currency=Currency.USD),
    Stock(ticker="TTE.PA", quantity=90, avg_cost=54.21, dividend=0.85*4, currency=Currency.EUR),
    Stock(ticker="MSFT", quantity=9, avg_cost=482.34, dividend=0.83*4, currency=Currency.USD),
    Stock(ticker="NVDA", quantity=35, avg_cost=159, dividend=0.01*4, currency=Currency.USD),
    Stock(ticker="BATS.L", quantity=100, avg_cost=3468.90, dividend=0.60*4, currency=Currency.GBP),
]