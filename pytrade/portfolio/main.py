
from pytrade.portfolio.optimize import OptimizePortfolio
from pytrade.portfolio import PORTFOLIO

if __name__ == "main":

    optimizer = OptimizePortfolio(PORTFOLIO)
    bounds = {
        'NVO': 0.075,
        "VWCE.DE": 1.0,
        "SCHD": 0.50,
        'NQSE.DE': 1.0,
        "QQQI": 0.10,
        'MSFT': 0.075,
        "AAPL": 0.075,
        "GOOG": 0.075
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