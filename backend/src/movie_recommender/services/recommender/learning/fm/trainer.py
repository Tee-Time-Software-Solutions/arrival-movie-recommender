from __future__ import annotations

import json
import pickle

from lightfm import LightFM
from tqdm import trange

from movie_recommender.services.recommender.paths_dev import ARTIFACTS
from movie_recommender.services.recommender.learning.fm.data import (
    load_lightfm_data,
)


MODEL_PATH = ARTIFACTS / "fm_lightfm_model.pkl"
MODEL_INFO_PATH = ARTIFACTS / "fm_lightfm_model_info.json"


NO_COMPONENTS = 32
EPOCHS = 15
NUM_THREADS = 4


def train_fm() -> None:
    """
    Train a LightFM model with BPR loss using the FM data artifacts.
    """
    print("Loading LightFM data (interactions + item features)...")
    interactions, item_features, mappings = load_lightfm_data()

    num_users, num_items = interactions.shape
    print(f"LightFM interactions: users={num_users}, items={num_items}")

    model = LightFM(
        no_components=NO_COMPONENTS,
        loss="bpr",
    )

    print("Training LightFM (BPR)...")
    # LightFM's `fit()` doesn't expose a progress callback; we train one epoch at a time
    # via `fit_partial()` to provide a visible progress bar.
    for _ in trange(EPOCHS, desc="LightFM epochs", unit="epoch"):
        model.fit_partial(
            interactions,
            item_features=item_features,
            epochs=1,
            num_threads=NUM_THREADS,
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Saving LightFM model and metadata...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    model_info = {
        "no_components": NO_COMPONENTS,
        "epochs": EPOCHS,
        "num_threads": NUM_THREADS,
        "num_users": int(num_users),
        "num_items": int(num_items),
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=4)

    print("LightFM model training complete and saved.")


if __name__ == "__main__":
    train_fm()

