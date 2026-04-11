import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from movie_recommender.services.recommender.paths_dev import (
    DATA_PROCESSED,
    artifacts_dir,
)


@dataclass(frozen=True)
class RecommenderArtifacts:
    movie_embeddings: np.ndarray
    user_embeddings: np.ndarray
    user_id_to_index: dict[int, int]
    movie_id_to_index: dict[int, int]
    index_to_movie_id: dict[int, int]
    movie_id_to_title: dict[int, str]


def _ensure_artifact_paths_exist() -> None:
    root = artifacts_dir()
    movie_emb = root / "movie_embeddings.npy"
    user_emb = root / "user_embeddings.npy"
    mappings = root / "mappings.json"
    movies_filtered = DATA_PROCESSED / "movies_filtered.parquet"
    required_paths = [
        movie_emb,
        user_emb,
        mappings,
        movies_filtered,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing recommender artifacts. Run the offline pipeline first.\n"
            f"{missing_list}"
        )


def load_recommender_artifacts() -> RecommenderArtifacts:
    _ensure_artifact_paths_exist()

    root = artifacts_dir()
    movie_embeddings_path = root / "movie_embeddings.npy"
    user_embeddings_path = root / "user_embeddings.npy"
    mappings_path = root / "mappings.json"
    movies_filtered_path = DATA_PROCESSED / "movies_filtered.parquet"

    movie_embeddings = np.load(movie_embeddings_path)
    user_embeddings = np.load(user_embeddings_path)

    with open(mappings_path, "r") as file:
        mappings = json.load(file)

    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {
        int(k): int(v) for k, v in mappings["movie_id_to_index"].items()
    }
    index_to_movie_id = {
        int(k): int(v) for k, v in mappings["index_to_movie_id"].items()
    }

    movies_df = pd.read_parquet(movies_filtered_path, columns=["movie_id", "title"])
    movie_id_to_title = {
        int(movie_id): str(title)
        for movie_id, title in zip(movies_df["movie_id"], movies_df["title"])
    }

    return RecommenderArtifacts(
        movie_embeddings=movie_embeddings,
        user_embeddings=user_embeddings,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id,
        movie_id_to_title=movie_id_to_title,
    )
