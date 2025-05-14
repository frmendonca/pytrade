import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    portfolio_value: float
    monthly_contributions: float
    returns_frequency: int
    option_iv_change: float = 0.0
    option_dte_change: int = 0


@dataclass
class SimulationResults:
    original_returns: np.ndarray
    transformed_returns: np.ndarray
