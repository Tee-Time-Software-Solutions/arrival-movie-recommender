from __future__ import annotations

import json
import pickle

from lightfm import LightFM
from tqdm import trange

from movie_recommender.services.recommender.utils.schema import Config
from movie_recommender.services.recommender.pipeline.offline.models.fm.steps.data import (
    load_lightfm_data,
)


def run(config: Config) -> None:
    """Train a LightFM model with BPR loss."""
    assets_dir = config.data_dirs.model_assets_dir
    fm = config.models.fm

    interactions, item_features, _ = load_lightfm_data(config)
    num_users, num_items = interactions.shape
    print(f"LightFM interactions: users={num_users}, items={num_items}")

    model = LightFM(no_components=fm.no_components, loss="bpr")

    print("Training LightFM (BPR)...")
    for _ in trange(fm.epochs, desc="LightFM epochs", unit="epoch"):
        model.fit_partial(
            interactions,
            item_features=item_features,
            epochs=1,
            num_threads=fm.num_threads,
        )

    with open(assets_dir / "fm_lightfm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(assets_dir / "fm_lightfm_model_info.json", "w") as f:
        json.dump(
            {
                "no_components": fm.no_components,
                "epochs": fm.epochs,
                "num_threads": fm.num_threads,
                "num_users": int(num_users),
                "num_items": int(num_items),
            },
            f,
            indent=4,
        )

    print("LightFM model training complete and saved.")
