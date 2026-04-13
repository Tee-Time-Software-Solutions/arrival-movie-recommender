from __future__ import annotations

import json
import pickle

import numpy as np
from implicit.bpr import BayesianPersonalizedRanking

from movie_recommender.services.recommender.utils.schema import Config
from movie_recommender.services.recommender.pipeline.offline.models.fm.steps.data import (
    load_lightfm_data,
)


def run(config: Config) -> None:
    """Train an implicit BPR matrix factorization model."""
    assets_dir = config.data_dirs.model_assets_dir
    fm = config.models.fm

    interactions, _, _ = load_lightfm_data(config)
    num_users, num_items = interactions.shape
    print(f"FM interactions: users={num_users}, items={num_items}")

    model = BayesianPersonalizedRanking(
        factors=fm.no_components,
        iterations=fm.epochs,
        num_threads=fm.num_threads,
    )

    print("Training implicit BPR...")
    # implicit expects a user×item sparse matrix (CSR works).
    model.fit(interactions)

    # Persist the fitted model for offline inspection/reuse.
    with open(assets_dir / "fm_bpr_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Export factors explicitly for serving-time dot-product scoring.
    # Shapes:
    # - user_factors: (num_users, factors)
    # - item_factors: (num_items, factors)
    np.save(assets_dir / "fm_user_factors.npy", model.user_factors)
    np.save(assets_dir / "fm_item_factors.npy", model.item_factors)

    with open(assets_dir / "fm_bpr_model_info.json", "w") as f:
        json.dump(
            {
                "model": "implicit.bpr.BayesianPersonalizedRanking",
                "factors": fm.no_components,
                "iterations": fm.epochs,
                "num_threads": fm.num_threads,
                "num_users": int(num_users),
                "num_items": int(num_items),
            },
            f,
            indent=4,
        )

    print("BPR model training complete and saved.")
