# For local development, we need to know the paths to the raw and processed data.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Precomputed paths
MOVIES_RAW = DATA_RAW / "movies.csv"
MOVIES_CLEAN = DATA_PROCESSED / "movies_clean.parquet"
INTERACTIONS_CLEAN = DATA_PROCESSED / "interactions_clean.parquet"
INTERACTIONS_FILTERED = DATA_PROCESSED / "interactions_filtered.parquet"
MOVIES_FILTERED = DATA_PROCESSED / "movies_filtered.parquet"