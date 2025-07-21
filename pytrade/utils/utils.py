import pandas as pd
import json
from typing import Union, Any, Callable
import numpy.typing as npt
from pathlib import Path



def check_all_same_sign(data: Union[pd.Series, npt.NDArray]):
    """
    :param data: data to be checked
    :return: boolean True if all same sign
    """
    return (data > 0).all() or (data < 0).all()



def read_json(path: Path) -> dict:
    """
    Reads a json file.
    :path: Path the path to the json file
    :returns parsed json in a dict format
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)



def df_factory(columns: list[str]) -> Callable[[list[tuple[Any]]], pd.DataFrame]:
    def factory(rows: list[tuple[Any]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=columns)

    return factory



def compute_stock_returns(df: pd.DataFrame, freq: int = 1):
    return (
        df[["Close"]]
        .assign(returns=df["Close"].pct_change(freq))
        .query("returns.notna()")
    )
