
import numpy as np
from scipy.optimize import minimize
from pytrade.simulation.utils import block_resample


class PerpetualWithdrawalRateFinder:
    def __init__(self):
        return None
    

    def compute_pwr(self, return_sequence: np.array) -> float:
        """
        Computes the Perpetual Withdrawal Rate as the maximum withdrawal rate
        that preserves initial capital given the 'return_sequence'

        params:
        return_sequence: np.array, the sequence of monthly returns

        returns the perpetual withdrawal rate estimate
        """

        optim = minimize(
            fun = self._optimize_pwr,
            x0 = (0.01),
            args = (return_sequence),
            method = "nelder-mead"
        )

        return optim.x

        
    @staticmethod
    def _optimize_pwr(pwr: float, return_sequence: np.array) -> float:
        rcum = np.prod((1 + return_sequence)*(1 - pwr / 12))
        return (10000*(rcum - 1))**2
