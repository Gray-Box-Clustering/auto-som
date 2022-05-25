from pydantic import BaseModel, HttpUrl, Field, FilePath
from api.utils.pymongo_utils import PyObjectId, ObjectId

from typing import Optional, Union, Literal


# class DatasetDescriptionSchema(BaseModel):
#     name: str
#     n_samples: int
#     n_features: int
#     url: Optional[HttpUrl] = None
#     feature_names: Optional[list[str]] = None


class DatasetDescriptionSchema(BaseModel):
    filepath_or_buffer: Union[HttpUrl, FilePath]
    names: Optional[list[str]] = None  # not necessary if dataset has names embedded
    usecols: tuple[int, int]  # max two features
    sep: Optional[str] = None
