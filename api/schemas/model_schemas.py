from pydantic import BaseModel, FilePath
from bson.binary import Binary
from typing import Any


class ModelSchema(BaseModel):
    model_name: str
    dataset_name: str
    trained_model: str
    # quantization_error: float
    feature_names: tuple[str, str]
    hyperparameters: dict
