from __future__ import annotations

import json
import pickle
from functools import lru_cache
from typing import Dict

import numpy as np

from movie_recommender.services.recommender.paths_dev import ARTIFACTS
from movie_recommender.services.recommender.learning.fm.data import (
    ITEM_FEATURES_PATH,
    MAPPINGS_PATH,
)


MODEL_PATH = ARTIFACTS / "fm_lightfm_model.pkl"


@lru_cache(maxsize=1)
def _load_lightfm_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_mappings() -> Dict[str, Dict[str, int]]:
    with open(MAPPINGS_PATH, "r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_item_features():
    from scipy.sparse import load_npz

    return load_npz(ITEM_FEATURES_PATH)


def score_user_movie(user_id: int, movie_id: int) -> float:
    """
    Compute LightFM score for a (user, movie) pair using the persisted model.
    """
    model = _load_lightfm_model()
    mappings = _load_mappings()
    item_features = _load_item_features()

    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {
        int(k): int(v) for k, v in mappings["movie_id_to_index"].items()
    }

    if user_id not in user_id_to_index or movie_id not in movie_id_to_index:
        return float(0.0)

    u_idx = user_id_to_index[user_id]
    i_idx = movie_id_to_index[movie_id]

    score = model.predict(
        user_ids=u_idx,
        item_ids=np.array([i_idx], dtype=np.int32),
        item_features=item_features,
    )[0]
    return float(score)


