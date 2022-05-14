from fastapi import APIRouter, Depends
from api.db import get_experiment_collection

router: APIRouter = APIRouter(
    prefix="/experiment",
    tags=["experiment"]
)


@router.get(
    path="/"
)
async def list_experiments(experiments=Depends(get_experiment_collection)):
    return experiments.find_one({}, {"_id": 0})
