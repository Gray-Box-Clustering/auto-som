from pydantic import BaseModel


class ContactSchema(BaseModel):
    name: str
    url: str
    email: str


class ApiDescriptionSchema(BaseModel):
    title: str
    version: str
    contact: ContactSchema
