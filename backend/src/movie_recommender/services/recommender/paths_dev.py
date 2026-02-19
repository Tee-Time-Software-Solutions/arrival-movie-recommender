# For local development, we need to know the paths to the raw and processed data.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
ARTIFACTS = PROJECT_ROOT / "artifacts"
