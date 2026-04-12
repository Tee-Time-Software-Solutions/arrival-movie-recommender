import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from movie_recommender.services.recommender.utils.schema import load_config


class RecommenderArtifacts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    movie_embeddings: np.ndarray
    user_embeddings: np.ndarray
    user_id_to_index: dict[int, int]
    movie_id_to_index: dict[int, int]
    index_to_movie_id: dict[int, int]
    movie_id_to_title: dict[int, str]
    all_movie_ids: np.ndarray


def load_model_artifacts() -> RecommenderArtifacts:
    config = load_config()
    assets_dir = config.data_dirs.model_assets_dir
    processed_dir = config.data_dirs.processed_dir

    required = [
        assets_dir / "movie_embeddings.npy",
        assets_dir / "user_embeddings.npy",
        assets_dir / "mappings.json",
        processed_dir / "movies_filtered.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing recommender artifacts. Run the offline pipeline first.\n"
            + "\n".join(f"- {p}" for p in missing)
        )

    movie_embeddings = np.load(assets_dir / "movie_embeddings.npy")
    user_embeddings = np.load(assets_dir / "user_embeddings.npy")

    with open(assets_dir / "mappings.json") as f:
        mappings = json.load(f)

    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {
        int(k): int(v) for k, v in mappings["movie_id_to_index"].items()
    }
    index_to_movie_id = {
        int(k): int(v) for k, v in mappings["index_to_movie_id"].items()
    }

    movies_df = pd.read_parquet(
        processed_dir / "movies_filtered.parquet", columns=["movie_id", "title"]
    )
    movie_id_to_title = {
        int(movie_id): str(title)
        for movie_id, title in zip(movies_df["movie_id"], movies_df["title"])
    }

    all_movie_ids = np.array(
        [index_to_movie_id[i] for i in range(len(index_to_movie_id))], dtype=np.int32
    )

    return RecommenderArtifacts(
        movie_embeddings=movie_embeddings,
        user_embeddings=user_embeddings,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id,
        movie_id_to_title=movie_id_to_title,
        all_movie_ids=all_movie_ids,
    )
