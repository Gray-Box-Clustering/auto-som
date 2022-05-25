from fastapi import APIRouter, Depends, status
from api.db import client_session, get_model_collection
from api.schemas.model_schemas import ModelSchema

from typing import Any

router = APIRouter(
    prefix="/model",
    tags=['model']
)


@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=str
)
async def register_model(
        model: ModelSchema,
        models: client_session = Depends(get_model_collection)
):
    """
    ## Register trained model
    """
    # model.trained_model = bytes(model.trained_model)
    model_dict: dict[str, Any] = dict(model)
    query = models.insert_one(model_dict)
    return str(query.inserted_id)
