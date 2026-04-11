from __future__ import annotations

import json
import pickle
from functools import lru_cache
from typing import Dict

import numpy as np
from scipy.sparse import load_npz

from movie_recommender.services.recommender.paths_dev import artifacts_dir


def _fm_model_path():
    return artifacts_dir() / "fm_lightfm_model.pkl"


def _fm_item_features_path():
    return artifacts_dir() / "fm_item_features.npz"


def _fm_mappings_path():
    return artifacts_dir() / "fm_mappings.json"


@lru_cache(maxsize=1)
def _load_lightfm_model():
    with open(_fm_model_path(), "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_mappings() -> Dict[str, Dict[str, int]]:
    with open(_fm_mappings_path(), "r") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def _get_processed_mappings() -> tuple[dict[int, int], dict[int, int]]:
    """
    Return mappings as int->int dicts, cached to avoid rebuilding per call.
    """
    mappings = _load_mappings()
    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {int(k): int(v) for k, v in mappings["movie_id_to_index"].items()}
    return user_id_to_index, movie_id_to_index


@lru_cache(maxsize=1)
def _load_item_features():
    return load_npz(_fm_item_features_path())


def score_user_movie(user_id: int, movie_id: int) -> float:
    """
    Compute LightFM score for a (user, movie) pair using the persisted model.
    """
    model = _load_lightfm_model()
    item_features = _load_item_features()

    user_id_to_index, movie_id_to_index = _get_processed_mappings()

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


