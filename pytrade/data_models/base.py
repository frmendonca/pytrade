import numpy.typing as npt
import pandas as pd
from dataclasses import dataclass
from typing import Any
from enum import StrEnum

@dataclass
class SimulationConfig:
    portfolio_value: float
    contributions: float
    returns_frequency: int
    option_dte_change: int = 0


@dataclass
class SimulationResults:
    original_returns: Any
    transformed_returns: Any


@dataclass
class PortfolioCash:
    currency: str
    amount: float


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
