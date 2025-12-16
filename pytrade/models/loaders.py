
import pathlib
from pytrade.models.serializer import from_json

MODEL_PATHS = pathlib.Path(__file__).parents[1].absolute() / "models"

def load_model(model_name: str):
    model_pipeline_path = MODEL_PATHS / model_name / "model_pipeline.json"
    ml_model = from_json(model_pipeline_path)
    model = ml_model.model
    inputs = ml_model.features
    return model, inputs


