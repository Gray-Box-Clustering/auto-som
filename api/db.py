from pymongo import MongoClient, client_session, errors
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


def get_dataset_collection():
    """ Yields dataset collection of autoclustering_db """
    session = client.start_session()
    try:
        dataset_collection = session.client.autoclustering_db.dataset
        dataset_collection.create_index("filepath_or_buffer", unique=True)
        dataset_collection.create_index("names")
        dataset_collection.create_index("usecols")
        dataset_collection.create_index("sep")
        yield dataset_collection
    finally:
        session.end_session()


def get_model_collection():
    """ Yields model collection of autoclustering_db """
    session = client.start_session()
    try:
        model_collection = session.client.autoclustering_db.model
        model_collection.create_index("model_name", unique=True)
        model_collection.create_index("dataset_name")
        model_collection.create_index("trained_model")
        model_collection.create_index("feature_names")
        yield model_collection
    finally:
        session.end_session()
