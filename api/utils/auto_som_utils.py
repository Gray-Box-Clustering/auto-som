import random
from typing import Any

import pandas as pd
import numpy as np
import requests

from api.utils.algorithms.auto_som.main import train_auto_som_model


def main():
    # Get dataset meta
    meta: dict[str, Any] = requests.get(url="http://localhost:8000/dataset/object_id/628447a5f1921144e0a5bbc9").json()
    # Train model
    hyperparams, path_to_model = train_auto_som_model(dataset_meta=meta, model_name="seeds_best_model")
    print(hyperparams)
    # Register model
    rsp = requests.post(
        url="http://localhost:8000/model/",
        json={
            "model_name": "seeds_best_model",
            "dataset_name": "628447a5f1921144e0a5bbc9",
            "trained_model": path_to_model,
            "feature_names": [meta["names"][meta["usecols"][0]], meta["names"][meta["usecols"][1]]],
            "hyperparameters": hyperparams
        }
    )
    print(rsp.status_code, rsp.json(), sep="\n")


if __name__ == "__main__":
    main()

    # ----------
    # search_space
    # dataset
    # n
    #
    # architectures = random.choices(search_space, n_architectures)
    # scores = estimate(architectures)
    # ensemble
    #
    # for architecture in architectures:
    #     # predictors = (architectures, scores)
    #     ensemble.train(architectures, scores)
    #     candidates = argmin(estimate(random.choices(search_space, n_architectures)), n_architectures)
    #     for candidate in candidates:
    #         expected_improvements.append(compute_expected_improvement(candidate))
    #     architectures = argmin(expected_improvements)
    #     score = estimate(candidate_architecture)
    #     if score >= threshold:
    #         return architectures
    # return argmin

