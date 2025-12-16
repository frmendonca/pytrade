
from dataclasses import dataclass
from typing import Any

@dataclass
class SimulationConfig:
    underlying_ticker: str
    returns_frequency: int
    portfolio_value: float
    hedge_allocation: float
    option_dte_change: int = 0


@dataclass
class SimulationResults:
    original_returns: Any
    transformed_returns: Any
    hedge_cost: float
