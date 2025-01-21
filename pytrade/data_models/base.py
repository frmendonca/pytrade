
import numpy as np
from dataclasses import dataclass
from scipy.stats._distn_infrastructure import rv_continuous_frozen as RVContinuousFrozen


@dataclass
class SimulationConfig:
    portfolio_value: float
    monthly_contributions: float
    returns_frequency: int
    number_of_years: int

@dataclass
class SimulationResults:
    hedged_returns: np.array
    unhedged_returns: np.array
    vix_returns: np.array
