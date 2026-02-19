import pandas as pd
from pathlib import Path
from movie_recommender.services.recommender.paths_dev import DATA_PROCESSED

INTERACTIONS_CLEAN_PATH = DATA_PROCESSED / "interactions_clean.parquet"
INTERACTIONS_FILTERED_PATH = DATA_PROCESSED / "interactions_filtered.parquet"
MOVIES_CLEAN_PATH = DATA_PROCESSED / "movies_clean.parquet"
MOVIES_FILTERED_PATH = DATA_PROCESSED / "movies_filtered.parquet"

# Quick size overview for all processed files
print("=== DATASET SIZES (MB) ===")
for name, path in [
    ("interactions_clean", INTERACTIONS_CLEAN_PATH),
    ("interactions_filtered", INTERACTIONS_FILTERED_PATH),
    ("movies_clean", MOVIES_CLEAN_PATH),
    ("movies_filtered", MOVIES_FILTERED_PATH),
]:
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"{name}: {size_mb:.2f} MB")

# Load detailed subsets
interactions = pd.read_parquet(INTERACTIONS_FILTERED_PATH)
movies = pd.read_parquet(MOVIES_FILTERED_PATH)

# --- Interactions ---
print("\n=== INTERACTIONS (filtered) ===")
print("Shape:", interactions.shape)
print("Columns:", interactions.columns.tolist())
print("Dtypes:")
print(interactions.dtypes)
print("\nHead:")
print(interactions.head())

print("\nUnique users:", interactions["user_id"].nunique())
print("Unique movies:", interactions["movie_id"].nunique())

# --- Movies ---
print("\n=== MOVIES (filtered) ===")
print("Shape:", movies.shape)
print("Columns:", movies.columns.tolist())
print("Dtypes:")
print(movies.dtypes)
print("\nHead:")
print(movies.head())

print("\nUnique movie_ids:", movies["movie_id"].nunique())
