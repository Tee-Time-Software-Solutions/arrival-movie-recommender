from __future__ import annotations

import json
import pickle
from functools import lru_cache

import numpy as np
from scipy.sparse import load_npz

from movie_recommender.services.recommender.utils.schema import Config


def score_user_movie(user_id: int, movie_id: int, config: Config) -> float:
    """Compute LightFM score for a (user, movie) pair using the persisted model."""
    assets_dir = config.data_dirs.model_assets_dir

    with open(assets_dir / "fm_lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)

    item_features = load_npz(assets_dir / "fm_item_features.npz")

    with open(assets_dir / "fm_mappings.json") as f:
        mappings = json.load(f)

    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {
        int(k): int(v) for k, v in mappings["movie_id_to_index"].items()
    }

    if user_id not in user_id_to_index or movie_id not in movie_id_to_index:
        return 0.0

    score = model.predict(
        user_ids=user_id_to_index[user_id],
        item_ids=np.array([movie_id_to_index[movie_id]], dtype=np.int32),
        item_features=item_features,
    )[0]
    return float(score)
