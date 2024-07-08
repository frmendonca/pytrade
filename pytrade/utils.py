
import json
import pandas as pd
from pathlib import Path
from typing import Any, Callable

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
