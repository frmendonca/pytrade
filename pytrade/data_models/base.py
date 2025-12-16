import numpy.typing as npt
import pandas as pd
from dataclasses import dataclass
from typing import Any
from enum import StrEnum


@dataclass
class PortfolioCash:
    currency: str
    amount: float


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
