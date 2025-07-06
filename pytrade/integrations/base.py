
import pandas as pd
from dataclasses import dataclass


@dataclass
class Ticker:
    symbol: str
    price: pd.Series
    data: pd.DataFrame
