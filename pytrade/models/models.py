
from typing import Any
from enum import StrEnum
from dataclasses import dataclass

@dataclass(frozen=True)
class MLModel:
    """
    A dataclass representing a ML
    model
    """
    model_name: str
    model: Any
    features: list[str]


class ModelInstance(StrEnum):
    """
    Model instance type
    """
    LINEAR_REGRESSION = "linear_regression"