from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from movie_recommender.services.recommender.utils.schema import Config


def score_user_movie(user_id: int, movie_id: int, config: Config) -> float:
    """Compute dot-product score for a (user, movie) pair using exported BPR factors."""
    user_factors, item_factors, user_id_to_index, movie_id_to_index = _load_assets(
        config.data_dirs.model_assets_dir
    )

    if user_id not in user_id_to_index or movie_id not in movie_id_to_index:
        return 0.0

    u = user_factors[user_id_to_index[user_id]]
    v = item_factors[movie_id_to_index[movie_id]]
    return float(np.dot(u, v))


@lru_cache(maxsize=1)
def _load_assets(
    assets_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    user_factors = np.load(assets_dir / "bpr_user_factors.npy")
    item_factors = np.load(assets_dir / "bpr_item_factors.npy")

    with open(assets_dir / "bpr_mappings.json") as f:
        mappings = json.load(f)

    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {
        int(k): int(v) for k, v in mappings["movie_id_to_index"].items()
    }
    return user_factors, item_factors, user_id_to_index, movie_id_to_index
