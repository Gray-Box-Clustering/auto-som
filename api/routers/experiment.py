from fastapi import APIRouter, Depends
from api.db import get_experiment_collection

# from api.utils.algorithms.auto_som.main import load_from_url, datasets
from typing import Any
import pandas as pd

router: APIRouter = APIRouter(
    prefix="/experiment",
    tags=["experiment"]
)


@router.get(
    path="/"
)
async def list_experiments(experiments=Depends(get_experiment_collection)):
    # params: dict[str, Any] = {
    #     "filepath_or_buffer": datasets["seeds_dataset"]["url"],
    #     "names": datasets["seeds_dataset"]["features"],
    #     "usecols": [0, 5],
    #     "sep": "\t+",
    #
    # }
    # seeds: pd.DataFrame = load_from_url(**params)
    # print(seeds)
    query: dict[str, str] = {
        "_id": "exampleObjectID",
        "name": "SomeVeryBasicName",
        "contents": "Some very basic contents"
    }
    return experiments.find_one({}, {"_id": 0})


@router.post(
    path="/create"
)
async def create_experiment(
        dataset_path_or_url
):
    ...
