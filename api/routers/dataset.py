from fastapi import APIRouter, Depends, status, HTTPException
from api.schemas.dataset_schemas import DatasetDescriptionSchema
from api.db import client_session, get_dataset_collection, errors
from api.utils.pymongo_utils import PyObjectId

from typing import Any

router = APIRouter(
    prefix="/dataset",
    tags=["dataset"]
)


@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=str
)
async def register_dataset(
        dataset_meta: DatasetDescriptionSchema,
        datasets: client_session = Depends(get_dataset_collection)
):
    """
    ## Provide dataset description
    """
    dataset_meta_dict: dict[str, Any] = dict(dataset_meta)
    try:
        query = datasets.insert_one(dataset_meta_dict)
        return str(query.inserted_id)
    except errors.DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset with filepath_or_buffer: {dataset_meta.filepath_or_buffer} already exists"
        )


@router.get(
    path="/",
    response_model=list[DatasetDescriptionSchema]
)
async def get_registered_datasets(datasets: client_session = Depends(get_dataset_collection)):
    """
    ## List registered datasets
    """
    items = list(datasets.find())
    if items:
        return [DatasetDescriptionSchema(**d) for d in items]
    return []


@router.get(
    path="/object_id/{object_id}",
    response_model=DatasetDescriptionSchema
)
async def get_registered_dataset(
        object_id: PyObjectId,
        datasets: client_session = Depends(get_dataset_collection)
):
    """
    ## Get registered dataset by its object id
    """
    query: dict[str, Any] = {
        "_id": PyObjectId(object_id)
    }
    dataset = datasets.find_one(query)
    if dataset:
        return DatasetDescriptionSchema(**dataset)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Dataset with ObjectId {object_id} doesn't exist"
    )
