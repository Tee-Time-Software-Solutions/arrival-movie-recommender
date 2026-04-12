import json

import numpy as np
from scipy.sparse import load_npz
import implicit

from movie_recommender.services.recommender.utils.schema import Config


def run(config: Config) -> None:
    assets_dir = config.data_dirs.model_assets_dir
    als = config.models.als

    print("Loading sparse matrix...")
    R_train = load_npz(assets_dir / "R_train.npz")

    print("Converting to confidence matrix...")
    C = R_train.copy()
    C.data = 1 + als.alpha * np.abs(C.data)

    print("Training implicit ALS model...")
    model = implicit.als.AlternatingLeastSquares(
        factors=als.factors,
        regularization=als.regularization,
        iterations=als.iterations,
        use_gpu=False,
    )
    model.fit(C)

    movie_embeddings = model.item_factors.astype(np.float32)
    user_embeddings = model.user_factors.astype(np.float32)
    print(
        f"Movie embeddings: {movie_embeddings.shape}, User embeddings: {user_embeddings.shape}"
    )

    np.save(assets_dir / "movie_embeddings.npy", movie_embeddings)
    np.save(assets_dir / "user_embeddings.npy", user_embeddings)

    with open(assets_dir / "model_info.json", "w") as f:
        json.dump(
            {
                "factors": als.factors,
                "regularization": als.regularization,
                "iterations": als.iterations,
                "alpha": als.alpha,
                "num_movies": movie_embeddings.shape[0],
                "num_users": user_embeddings.shape[0],
                "embedding_dim": movie_embeddings.shape[1],
            },
            f,
            indent=4,
        )

    print("ALS training complete. Artifacts saved.")
