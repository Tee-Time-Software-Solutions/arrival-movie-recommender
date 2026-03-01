import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from movie_recommender.services.recommender.paths_dev import ARTIFACTS, DATA_PROCESSED


MOVIE_EMBEDDINGS_PATH = ARTIFACTS / "movie_embeddings.npy"
USER_EMBEDDINGS_PATH = ARTIFACTS / "user_embeddings.npy"
MAPPINGS_PATH = ARTIFACTS / "mappings.json"
MOVIES_FILTERED_PATH = DATA_PROCESSED / "movies_filtered.parquet"


@dataclass(frozen=True)
class RecommenderArtifacts:
    movie_embeddings: np.ndarray
    user_embeddings: np.ndarray
    user_id_to_index: dict[int, int]
    movie_id_to_index: dict[int, int]
    index_to_movie_id: dict[int, int]
    movie_id_to_title: dict[int, str]


def _ensure_artifact_paths_exist() -> None:
    required_paths = [
        MOVIE_EMBEDDINGS_PATH,
        USER_EMBEDDINGS_PATH,
        MAPPINGS_PATH,
        MOVIES_FILTERED_PATH,
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

    movie_embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
    user_embeddings = np.load(USER_EMBEDDINGS_PATH)

    with open(MAPPINGS_PATH, "r") as file:
        mappings = json.load(file)

    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {
        int(k): int(v) for k, v in mappings["movie_id_to_index"].items()
    }
    index_to_movie_id = {int(k): int(v) for k, v in mappings["index_to_movie_id"].items()}

    movies_df = pd.read_parquet(MOVIES_FILTERED_PATH, columns=["movie_id", "title"])
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
