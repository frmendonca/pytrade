

import pandas as pd
import numpy as np
import pathlib
from pytrade.models.loaders import load_model


def model_predict(df: pd.DataFrame) -> pd.Series:

    model_name = pathlib.Path(__file__).parent.name
    model, inputs = load_model(model_name)

    return np.exp(model.predict(df[inputs])) - 1