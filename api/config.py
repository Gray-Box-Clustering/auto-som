from pydantic import BaseSettings


class MongoSettings(BaseSettings):

    MONGO_CONTAINER: str
    MONGO_USERNAME: str
    MONGO_PASSWORD: str

    class Config:
        env_file = "env/db.env"


mongo_settings = MongoSettings()
# TODO: make sure env vars override
# mongo_settings = MongoSettings(_env_file="env/db.env")
