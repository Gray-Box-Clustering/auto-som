import bson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from minisom import MiniSom
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from typing import Optional, Literal, Any
# from sklearn.cluster import KMeans
from numpy.linalg import norm

import pickle
import base64
import os
from bson import Binary, BSON
from datetime import datetime


class Data:

    @staticmethod
    def load_from_url(verbose: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load data from URL.

        * As of now, loading data from local source is not supported due to collisions.

        :param verbose: bool. Flash DataFrame head when loaded
        :param kwargs: dict. Any of pandas.read_csv() parameters
        :return: pd.DataFrame
        """
        # FIXME: data collision when loading from local file
        data: pd.DataFrame = pd.read_csv(engine='python', **kwargs)
        print(data.head()) if verbose else ...
        return data

    @staticmethod
    def save() -> None:
        raise NotImplementedError

    @staticmethod
    def normalize(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Data normalization (required for minisom to run properly).

        :param data: pd.DataFrame. Loaded by Data.load_from_url() 2-dimensional data
        :param verbose: bool. Flash DataFrame head when normalized
        :return: pd.DataFrame
        """
        normalized: pd.DataFrame = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        print(normalized.head()) if verbose else ...
        return normalized

    @staticmethod
    def plot(x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """
        Plot 2-dimensional data to explore it.

        :param x: pd.DataFrame. Single DataFrame column
        :param y: pd.DataFrame. Single DataFrame column
        :param kwargs: dict. Any of plt.scatter() parameters
        :return: None
        """
        plt.scatter(np.array(x), np.array(y), **kwargs)
        plt.show()


class Network:

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Self Organizing Map initialization

        :param data: pd.DataFrame. Normalized 2-d data via Data.normalize()
        :param kwargs: dict. Any of MiniSom() parameters
        """
        self.data: pd.DataFrame = data

        self.x: int = int(np.sqrt(5 * np.sqrt(self.data.shape[0])))
        self.y: int = self.data.shape[1]
        self.input_len: int = self.data.shape[1]

        self.neighborhood_function: str = kwargs.get("neighborhood_function", "gaussian")
        self.activation_distance: str = kwargs.get("activation_distance", "euclidean")
        self.sigma: float = kwargs.get("sigma", .5)
        self.learning_rate: float = kwargs.get("learning_rate", .5)
        self.random_seed: int = kwargs.get("random_seed", 10)

        self.params: dict = {
            "x": self.x,
            "y": self.y,
            "input_len": self.input_len,
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "neighborhood_function": self.neighborhood_function,
            "activation_distance": self.activation_distance,
            "random_seed": self.random_seed
        }

        self.net: MiniSom = MiniSom(**self.params)

        print("Network with following parameters was created", self.params)

    def get_winner_neuron_coord(self, verbose: bool = False) -> np.array:
        winner_neuron_coord: np.array = np.array([self.net.winner(x) for x in np.array(self.data)]).T
        print(winner_neuron_coord) if verbose else ...
        return winner_neuron_coord  # bi-dimensional

    def get_cluster_index(self, verbose: bool = False) -> np.array:
        cluster_idx: np.array = np.ravel_multi_index(self.get_winner_neuron_coord(), (self.x, self.y))
        print(cluster_idx) if verbose else ...
        return cluster_idx  # mono-dimensional

    def train(self, data: np.array, n_iters: int = 500) -> None:
        self.net.train_random(data=data, num_iteration=n_iters)

    def plot_clusters(self, data: np.array, cluster_idx: np.array) -> None:
        # Clusters
        for c in np.unique(cluster_idx):
            plt.scatter(
                data[cluster_idx == c, 0],
                data[cluster_idx == c, 1],
                label='cluster=' + str(c),
                alpha=.7
            )
        # Centroids/neurons
        for centroid in self.net.get_weights():
            plt.scatter(
                centroid[:, 0], centroid[:, 1], marker='x',
                s=10, linewidths=15, color='k', label='centroid'
            )
        plt.legend()
        plt.show()


class SearchSpace:

    """
    Search Space consists of BASE and SPACE.

    BASE is a set of default hyperparameters for network initialization.
    SPACE is a set of ranges Network hyperparameters may be initialized with during Search process.
    """

    BASE: dict = {
        "neighborhood_function": "gaussian",
        "activation_distance": "euclidean",
        "sigma": .5,
        "learning_rate": .5
    }

    SPACE: dict = {
        "neighborhood_function": hp.choice("neighborhood_function", [
            'gaussian', 'mexican_hat', 'bubble', 'triangle'
        ]),
        "activation_distance": hp.choice("activation_distance", ['euclidean', 'cosine', 'manhattan', 'chebyshev']),
        "sigma": hp.uniform("sig", 0.001, 1.00),
        "learning_rate": hp.uniform("learning_rate", 0.001, 5.0)
    }


class SearchStrategy:

    @staticmethod
    def random_search(net: Network, dataset: np.array, epochs: int = 15, n_iters: int = 500) -> dict:

        x: int = net.x
        y: int = net.y
        input_len: int = net.input_len

        def tune(space: dict) -> dict:
            sig = space["sigma"]
            learning_rate = space["learning_rate"]
            neighborhood_function = space["neighborhood_function"]
            activation_distance = space["activation_distance"]
            value = MiniSom(
                x, y, input_len, sig, learning_rate, neighborhood_function, activation_distance=activation_distance
            ).quantization_error(dataset)
            return {
                "loss": value,
                "status": STATUS_OK
            }

        architectures: dict = dict()

        for epoch in range(epochs):
            trials: Trials = Trials()
            best: dict = fmin(fn=tune, space=SearchSpace.SPACE, algo=tpe.suggest, max_evals=500, trials=trials)
            print(f"best: {best}")
            for i, trial in enumerate(trials.trials[:2]):
                print(i, trial)
            sigma = best["sig"]
            learning_rate = best["learning_rate"]
            neighbors: list = ['gaussian', 'mexican_hat', 'bubble', 'triangle']
            neighborhood_function = neighbors[best['neighborhood_function']]
            distances: list = ['euclidean', 'cosine', 'manhattan', 'chebyshev']
            activation_distance = distances[best['activation_distance']]
            print(
                f"x: {x} y: {y} input_len: {input_len} sigma: {sigma} learning_rate: {learning_rate} neighborhood_function: {neighborhood_function} activation_distance: {activation_distance}"
            )

            som: Network = Network(
                data=dataset,
                sigma=sigma,
                lr=learning_rate,
                neighborhood_function=neighborhood_function,
                activation_distance=activation_distance
            )

            som.net.random_weights_init(dataset)
            # som.net.pca_weights_init(dataset)
            som.net.train_random(dataset, n_iters, verbose=True)

            error: float = Estimator.compute_quantization_error(net=som.net, data=dataset, verbose=True)
            # TODO: write early stop
            architectures[error] = {
                "x": x,
                "y": y,
                "input_len": input_len,
                "sigma": sigma,
                "learning_rate": learning_rate,
                "neighborhood_function": neighborhood_function,
                "activation_distance": activation_distance
            }

            print(architectures[error])

            winner_coordinates = np.array([som.net.winner(x) for x in dataset]).T
            cluster_index = np.ravel_multi_index(winner_coordinates, (x, y))

            som.plot_clusters(data=dataset, cluster_idx=cluster_index)  # plots are stored in 'plots/'
            
            # ##############################
            # with open(f"{str(datetime.utcnow())}_som.p", 'wb') as outfile:
            #     pickle.dump(som, outfile)
            # #############################

            if error < Estimator.THRESHOLD:
                return architectures
        return architectures


class Estimator:

    THRESHOLD: float = 0.1  # When quantization error hits this threshold search process can be stopped

    @staticmethod
    def compute_quantization_error(net: MiniSom, data: np.array, verbose: bool = False) -> float:
        """
        Compute Quantization Error

        Average distance of the sample vectors to the cluster centroids (neurons). Should be as low as possible

        https://stackoverflow.com/a/48196135

        * NOTE: Can't use Quantization Error comparing networks of different sizes
        (x and y should stay the same throughout experiments)

        :param net: MiniSom. Trained model
        :param data: np.array
        :param verbose: bool. Flash resulting value
        :return: float
        """
        quantization_error: float = net.quantization_error(data)
        print(quantization_error) if verbose else ...
        return quantization_error

    @staticmethod
    def compute_quantization_error_manually(data: np.array, centroids: np.array) -> float:
        """
        Same as compute_quantization_error() above but not for instances of MiniSom

        :param data: np.array. 2-d data points
        :param centroids: np.array. Same size as data where each entry corresponds to data point
        :return: float
        """
        return norm(data - centroids, axis=1).mean()


class Experiment:
    """  """

    # @staticmethod
    # def kmeans_elbow(data: np.array) -> None:
    #     """
    #     Computes elbow method to explore optimal n_clusters parameter
    #     :param data: np.array. Normalized 2-d data
    #     :return: None
    #     """
    #     wcss: list = []
    #     for i in range(1, 11):
    #         km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #         km.fit(data)
    #         wcss.append(km.inertia_)
    #     plt.plot(range(1, 11), wcss)
    #     plt.title('Elbow Method')
    #     plt.xlabel('Number of clusters')
    #     plt.ylabel('wcss')
    #     plt.show()

    # @staticmethod
    # def kmeans(
    #         data: np.array, n_clusters: int, init: str = 'k-means++', max_iter: int = 300, n_init: int = 10,
    #         random_state: int = 0
    # ) -> tuple[KMeans, np.array]:
    #     km = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
    #     y_means = km.fit_predict(data)
    #     for cl in range(n_clusters):
    #         plt.scatter(data[y_means == cl, 0], data[y_means == cl, 1], s=50)
    #     plt.scatter(
    #         km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=200, marker='s', c='red', alpha=0.7,
    #         label='Centroids'
    #     )
    #     plt.legend()
    #     plt.show()
    #     return km, y_means


def train_auto_som_model(dataset_meta: dict[str, Any], model_name: str):
    """  """
    # 1. Load dataset
    dataset: pd.DataFrame = Data.load_from_url(**dataset_meta)
    # 2. Normalize data
    dataset_normalized: pd.DataFrame = Data.normalize(dataset)
    # 3. Initialize base network model
    som: Network = Network(dataset_normalized)
    # 4. Train base network model
    dataset_array: np.array = np.array(dataset_normalized)
    n_iters: int = 500
    som.train(data=dataset_array, n_iters=n_iters)
    dataset_error: float = Estimator.compute_quantization_error(net=som.net, data=dataset_array)
    dataset_winner_neuron: np.array = som.get_winner_neuron_coord()
    dataset_cluster_index: np.array = som.get_cluster_index()
    # 5. Search
    dataset_architectures: dict = SearchStrategy.random_search(som, dataset_array)
    dataset_min_error: float = min(dataset_architectures)
    dataset_best_arch: dict = dataset_architectures[dataset_min_error]
    dataset_best_arch.pop("x")
    dataset_best_arch.pop("y")
    dataset_best_arch.pop("input_len")
    # 6. Train best
    best_som: Network = Network(dataset_normalized, **dataset_best_arch)
    best_som.train(data=dataset_array, n_iters=n_iters)
    # 7. Save best model
    model_path: os.path = os.path.join(os.path.join(os.path.dirname(__file__), f"models/{model_name}.p"))
    with open(model_path, "wb+") as outfile:
        pickle.dump(best_som.net, outfile)
    print("Model saved to:" + model_path)
    return dataset_best_arch, os.path.abspath(model_path)


def load_model_and_predict(som: Network):
    # with open('som.p', 'rb') as infile:
    #     som = pickle.load(infile)
    ...


# def example():
# seeds_dataset_meta = requests.get(url="http://localhost:8000/dataset/object_id/628447a5f1921144e0a5bbc9").json()
# seeds_best_model = train_auto_som_model(seeds_dataset_meta)
# cd into auto_som directory before running script
# TODO: store path to model instead of model
# model_path: os.path = os.path.join(os.path.join(os.path.dirname(__file__), "models/seeds_best_model.p"))
# with open(model_path, "wb+") as outfile:
#     pickle.dump(seeds_best_model.net, outfile)


# with open('models/seeds_best_model.p', 'rb') as infile:
#     seeds_best_model = pickle.load(infile)
#     # with open('models/seeds_best_model.p', 'rb') as infile:
#     #     seeds_best_model = infile.read()
#     # print(seeds_best_model.__dict__)
#     encoded = base64.b64encode(seeds_best_model)
# rsp = requests.post(
#     url="http://localhost:8000/model/",
#     json={
#         "model_name": "seeds_best_model",
#         "dataset_name": "628447a5f1921144e0a5bbc9",
#         "trained_model": model_path,
#         # "trained_model": bson.Binary(pickle.dumps(seeds_best_model)),
#         # "trained_model": encoded,
#         # "trained_model": bson.Binary(infile.read()),
#         # "trained_model": bson.encode(seeds_best_model),
#         "feature_names": [
#             "area",
#             "asymmetry_coefficient"
#         ]
#     }
# )
# print(rsp.status_code, rsp.json())
