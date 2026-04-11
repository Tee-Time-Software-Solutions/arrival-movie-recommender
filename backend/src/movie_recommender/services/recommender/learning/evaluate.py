import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from movie_recommender.services.recommender.paths_dev import DATA_SPLITS, artifacts_dir
from movie_recommender.services.recommender.learning.metrics import dcg_at_k

TRAIN_PATH = DATA_SPLITS / "train.parquet"
VAL_PATH = DATA_SPLITS / "val.parquet"

K = 10

def evaluate():
    root = artifacts_dir()
    user_emb_path = root / "user_embeddings.npy"
    movie_emb_path = root / "movie_embeddings.npy"
    mappings_path = root / "mappings.json"

    print("Loading embeddings...")
    user_embeddings = np.load(user_emb_path)
    movie_embeddings = np.load(movie_emb_path)

    print("Loading mappings...")
    with open(mappings_path, "r") as f:
        mappings = json.load(f)

    user_id_to_index = {int(k): v for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {int(k): v for k, v in mappings["movie_id_to_index"].items()}
    index_to_movie_id = {int(k): v for k, v in mappings["index_to_movie_id"].items()}

    print("Loading splits...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    print("Building lookup tables...")
    train_lookup = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    val_lookup = val_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    recall_scores = []
    precision_scores = []
    ndcg_scores = []

    print("Evaluating...")

    for user_id in tqdm(val_lookup.keys()):
        if user_id not in user_id_to_index:
            continue

        user_index = user_id_to_index[user_id]

        user_vector = user_embeddings[user_index]

        # Compute scores
        scores = movie_embeddings @ user_vector

        # Remove seen movies
        seen_movies = train_lookup.get(user_id, set())
        seen_indices = [
            movie_id_to_index[m] for m in seen_movies if m in movie_id_to_index
        ]
        scores[seen_indices] = -np.inf

        # Get top-K
        top_k_indices = np.argpartition(scores, -K)[-K:]
        top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

        recommended_movies = {index_to_movie_id[idx] for idx in top_k_indices}

        true_movies = val_lookup[user_id]

        hits = recommended_movies & true_movies

        # Metrics
        recall = len(hits) / len(true_movies)
        precision = len(hits) / K

        relevance = [
            1 if index_to_movie_id[idx] in true_movies else 0 for idx in top_k_indices
        ]

        dcg = dcg_at_k(relevance)
        idcg = dcg_at_k(sorted(relevance, reverse=True))
        ndcg = dcg / idcg if idcg > 0 else 0

        recall_scores.append(recall)
        precision_scores.append(precision)
        ndcg_scores.append(ndcg)

    print("\n=== Evaluation Results ===")
    print(f"Recall@{K}:    {np.mean(recall_scores):.4f}")
    print(f"Precision@{K}: {np.mean(precision_scores):.4f}")
    print(f"NDCG@{K}:      {np.mean(ndcg_scores):.4f}")


if __name__ == "__main__":
    evaluate()
