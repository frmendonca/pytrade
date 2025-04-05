import pandas as pd

from typing import Union
import numpy.typing as npt


def check_all_same_sign(data: Union[pd.Series, npt.NDArray]):
    """
    :param data: data to be checked
    :return: boolean True if all same sign
    """
    return (data > 0).all() or (data < 0).all()
