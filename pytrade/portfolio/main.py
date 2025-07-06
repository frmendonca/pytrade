
from pytrade.portfolio.optimize import OptimizePortfolio
from pytrade.portfolio.config import PORTFOLIO

if __name__ == "main":

    optimizer = OptimizePortfolio(PORTFOLIO)
    bounds = {
        'NVO': 0.075,
        "VWCE.DE": 0.50,
        "SCHD": 0.20,
        'SXRV.DE': 0.10,
        'TTE.PA': 0.05,
        'MSFT': 0.075,
        "BATS.L": 0.05,
        "NVDA": 0.075,
        "AAPL": 0.075
    }

    res = optimizer.optimize_portfolio(
        bounds = bounds,
        objective = "ms"
    )

    weights = round(res["weights"], 5)
    solution_fn = res["objective"]

    print("Solution")
    print(weights)
    print("\nObjetive function")
    print(solution_fn)