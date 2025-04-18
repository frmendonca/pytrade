
import numpy as np

import numpy.typing as npt
from scipy.optimize import minimize, basinhopping
from scipy.stats import genpareto
from pytrade.integrations.yfinance_data import YFinanceClient
from typing import Literal

class Portfolio:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols

        # Load data
        data_client = YFinanceClient()
        self.data = data_client.fetch_data(self.symbols)
        self.data = self.data[self.symbols] # Enforce same order as in instanciated list


    def fit(self):
        """
        Optimize portfolio weights by minimizing the tail index
        of the portfolio
        :return: self
        """

        weights_init = np.ones(len(self.symbols))
        weights_init = weights_init / weights_init.sum() # Normalize to 1

        optim = minimize(
            fun=self._objective_function,
            x0 = weights_init,
            method="Nelder-Mead",
            args = ("ms"),
            options = {"maxiter": 5000}
        )
        weights_init = optim.x

        solution = self._normalize_weights(optim.x)
        solution_r = self.data.dot(solution)
        return {
            "weights": solution,
            "returns": solution_r,
            "tail_index": self._get_tail_index(solution_r),
            "bev": self._get_bev(solution_r),
            "optim": optim
        }


    @staticmethod
    def _normalize_weights(weights: npt.NDArray):
        weights = np.abs(weights)
        weights = weights / weights.sum()
        return weights


    def _objective_function(self, weights: npt.NDArray, obj_type: Literal["ms","tail"] = "ms"):
        """
        Computes the objective function to be minimized
        :return:
        """
        weights = self._normalize_weights(weights)
        r = self.data.dot(weights)
        neg_r = r[r<0]

        match obj_type:
            case "ms":
                return np.max(neg_r**4)/np.sum(neg_r**4)
            case "tail":
                return self._get_tail_index(r)


    @staticmethod
    def _get_tail_index(r: npt.NDArray) -> float:
        """
        Estimates tail index
        :param r: a sequence of returns
        :return: xi a float corresponding to the estimated tail index
        """

        # Estimate left tail index through MLE
        neg_returns = np.abs(r[r < 0])

        threshold = np.quantile(neg_returns, 0.95)
        exceedences = neg_returns[neg_returns > threshold] - threshold
        xi, loc, scale = genpareto.fit(exceedences, floc = 0) # Forced location to 0

        return xi


    @staticmethod
    def _get_bev(r: npt.NDArray) -> float:
        """
        Computes Bayes EV for a sequence of returns
        :param r: sequence of returns
        :return: BEV a float
        """
        return np.exp(np.mean(np.log(1 + r))) - 1

