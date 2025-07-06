import typing as t
import pandas as pd
import numpy as np
from pytrade.portfolio.base import unpack_portfolio

import scipy.optimize as opt

from pytrade.portfolio.config import PORTFOLIO


class OptimizePortfolio:

    def __init__(self, config: dict[str, list[t.Any]]):
        self.config = config
        self.raw = unpack_portfolio(self.config) # unpack portfolio        


    def optimize_portfolio(
        self, bounds: dict[str, float],
        objective: t.Literal["ms", "bev"] = "bev"
    ):     

        # Combine returns into one dataframe
        return_df = pd.DataFrame(
            {
                ticker: v["data"]["Returns"]
                for ticker, v in self.raw.items()
            }
        )
        return_df.dropna(axis = 0, inplace=True)
        initial_values = pd.Series([1.0/return_df.shape[1] for _ in range(return_df.shape[1])], index = return_df.columns)


        optimizer = opt.minimize(
            fun = self._loss_function,
            x0 = initial_values,
            method='SLSQP',
            args=(return_df, objective),
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            bounds = opt.Bounds(
                [0]*return_df.shape[1],
                [bounds.get(ticker, 1) for ticker in return_df.columns]
            )
        )

        if optimizer.status == 0:
            return {
                "weights": pd.Series(optimizer.x, index = return_df.columns),
                "objective": -optimizer.fun,
                "optim_obj": optimizer
            }
        else:
            RuntimeWarning("Optimization failed")
    

    def _loss_function(self, weights: pd.Series, return_df: pd.DataFrame, objective: t.Literal["ms", "bev"]) -> float:
        weights = pd.Series(weights, index = return_df.columns)

        weighted_returns = self._compute_weighted_returns(return_df, weights)

        if objective == "ms":
            left_tail_ms_4 = self._compute_ms(weighted_returns[weighted_returns < 0], 4)
            right_tail_ms_4 = self._compute_ms(weighted_returns[weighted_returns > 0], 4)
            return left_tail_ms_4 - right_tail_ms_4
        

        elif objective == "bev":
            bev = self._compute_bev(weighted_returns)
            annual_bev = ((1 + bev)**252 - 1)
            return -annual_bev

    @staticmethod
    def _compute_ms(returns: pd.Series, k: int):
        x = np.abs(returns)
        return np.max(x**k) / np.sum(x**k)


    @staticmethod
    def _compute_weighted_returns(return_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
        return return_df @ weights

    @staticmethod
    def _compute_bev(returns: pd.Series) -> float:
        return np.exp(np.mean(np.log(1 + returns))) - 1

    @staticmethod
    def _compute_total_dividend_yield(raw: dict[str, t.Any]):
        
        market_value = 0
        total_dividends = 0
        for ticker_info in raw.values():

            market_value += ticker_info["quantity"]*ticker_info["price"]
            total_dividends += ticker_info["quantity"]*ticker_info["dividend"]

        return total_dividends / market_value





