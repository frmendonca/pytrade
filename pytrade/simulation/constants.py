from scipy.stats import norm
from dataclasses import dataclass
from typing import List

from pytrade.constants import NUMERIC_INFINITY


@dataclass(frozen=True)
class TailParameters:
    """Parameters for tail generator distribution"""

    s: float
    loc: float
    scale: float


@dataclass(frozen=True)
class NormParameters:
    """Parameters for a normal distribution"""

    loc: float
    scale: float


# Define left tail parameters
LEFT_TAIL: TailParameters = TailParameters(
    s=0.9274826457963914, loc=-0.003154933610007448, scale=0.027694808964929315
)

RIGHT_TAIL: TailParameters = TailParameters(
    s=0.5853417322532056, loc=-0.009161941839938244, scale=0.035561782229894215
)


MIXTURE_PARAMETER: float = 0.6376351467307362

# VIX/SPY Relation
SPY_BINS: List[float] = [-1.0, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, NUMERIC_INFINITY]
VIX_PARAMETERS: List[NormParameters] = [
    NormParameters(loc=1.9163255161794854, scale=1.305277180270236),
    NormParameters(loc=0.5895435471692753, scale=0.4716956239592801),
    NormParameters(loc=0.35642887293459585, scale=0.32591074196897313),
    NormParameters(loc=0.12687135373205516, scale=0.2088557954470283),
    NormParameters(loc=-0.05071102269302206, scale=0.13245657228917934),
    NormParameters(loc=-0.17053220780223977, scale=0.14051498820792832),
    NormParameters(loc=-0.24850747327886272, scale=0.132687957369612),
]

VIX_GENERATORS = [norm(loc=params.loc, scale=params.scale) for params in VIX_PARAMETERS]
