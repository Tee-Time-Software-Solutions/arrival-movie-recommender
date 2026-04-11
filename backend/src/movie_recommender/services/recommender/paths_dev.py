# For local development, we need to know the paths to the raw and processed data.
from __future__ import annotations

import os
from pathlib import Path

_RECOMMENDER_ROOT = Path(__file__).resolve().parent

DATA_RAW = _RECOMMENDER_ROOT / "data" / "raw"
DATA_PROCESSED = _RECOMMENDER_ROOT / "data" / "processed"
DATA_SPLITS = _RECOMMENDER_ROOT / "data" / "splits"

ENV_ARTIFACT_VERSION = "RECOMMENDER_ARTIFACT_VERSION"

# Legacy alias: fixed at import time. Prefer artifacts_dir() in new code.
PROJECT_ROOT = _RECOMMENDER_ROOT


def artifacts_dir() -> Path:
    """
    Directory for ALS embeddings, mappings, R_train.npz, model_info.json.

    If RECOMMENDER_ARTIFACT_VERSION is set (e.g. ``20260411a``), uses
    ``artifacts/<version>/`` so multiple trained models can coexist.
    Resolved per call so the env var can be set before pipeline steps.
    """
    version = os.getenv(ENV_ARTIFACT_VERSION, "").strip()
    base = _RECOMMENDER_ROOT / "artifacts"
    if version:
        return base / version
    return base
