

import copy
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from pytrade.models.models import MLModel, ModelInstance
from pytrade.utils.utils import read_json
import pathlib


JSON_STRUCTURE = {
    "model_name": "",
    "model_instance": None,
    "params": {},
    "features": [],
    "coefficients": {
        "intercept": None,
        "coefficients": []
    }
}


def serialize(model: MLModel) -> dict:
    """
    Serializes a model into a json
    dictionary
    """
    model_pipeline = copy.deepcopy(JSON_STRUCTURE)

    model_pipeline["model_name"] = model.model_name
    model_pipeline["params"] = model.model.get_params()
    model_pipeline["features"] = model.features

    if isinstance(model.model, LinearRegression):
        model_pipeline["model_instance"] = ModelInstance.LINEAR_REGRESSION.value
        model_pipeline["coefficients"]["intercept"] = model.model.intercept_
        model_pipeline["coefficients"]["coefficients"] = list(model.model.coef_)

    return model_pipeline



def desearialize(model_pipeline: dict) -> MLModel:
    """
    Rebuilds the model considering the
    model_pipeline.json
    """

    if model_pipeline["model_instance"] == ModelInstance.LINEAR_REGRESSION.value:

        model_class = LinearRegression
        model_class = model_class(**model_pipeline["params"])

        model_class.intercept_ = model_pipeline["coefficients"]["intercept"]
        model_class.coef_ = np.array(model_pipeline["coefficients"]["coefficients"])

    else:
        raise NotImplementedError("Model instance type not implemented yet")

    return MLModel(
        model_name=model_pipeline["model_name"],
        model=model_class,
        features=model_pipeline["features"]
    )


def to_json(model: MLModel, output_path: str = "") -> None:
    model_pipeline = serialize(model)
    with open(f'{output_path}/model_pipeline.json', 'w', encoding='utf-8') as file:
        json.dump(model_pipeline, file, ensure_ascii=False, indent=4)

    return None


def from_json(path: pathlib.Path) -> MLModel:
    model_pipeline = read_json(path)
    return desearialize(model_pipeline)




