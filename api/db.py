from pymongo import MongoClient
from api.config import mongo_settings

client: MongoClient = MongoClient(
    f"mongodb://{mongo_settings.MONGO_USERNAME}:{mongo_settings.MONGO_PASSWORD}@{mongo_settings.MONGO_CONTAINER}/"
)


def get_experiment_collection():
    """ Yields experiment collection of autoclustering_db """
    session = client.start_session()
    try:
        experiment_collection = session.client.autoclustering_db.experiment
        experiment_collection.create_index("experiment_id", unique=True)
        experiment_collection.create_index("neighborhood_function")
        experiment_collection.create_index("activation_distance")
        experiment_collection.create_index("sigma")
        experiment_collection.create_index("learning_rate")
        yield experiment_collection
    finally:
        session.end_session()
