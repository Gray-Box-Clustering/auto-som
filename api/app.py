from fastapi import FastAPI
# from ..auto_som.main import asd
from api.routers import experiment, dataset, model
from api.schemas.app_schemas import ApiDescriptionSchema

app = FastAPI(
    title="AutoClusteringAPI",
    version="0.1.0",
    contact={
        "name": "Renat Kyryllov",
        "url": "https://github.com/krllps/auto-clustering-api",
        "email": "renat.kyryllov@gmail.com"
    },
    docs_url="/api/docs"
)


@app.get(
    path="/api",
    tags=["root"],
    response_model=ApiDescriptionSchema
)
async def get_api():
    """
    ## Get API description
    """
    description: dict[str, str] = {
        "title": app.title,
        "version": app.version,
        "contact": app.contact
    }
    return ApiDescriptionSchema(**description)


app.include_router(experiment.router)
app.include_router(dataset.router)
app.include_router(model.router)
