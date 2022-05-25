from pydantic import BaseModel


class DatasetSchema(BaseModel):
    url: str
