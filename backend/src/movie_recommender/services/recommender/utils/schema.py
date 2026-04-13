from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

# Recommender root = one level above this utils/ folder
_RECOMMENDER_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _RECOMMENDER_ROOT / "config.yaml"


class DataConfig(BaseModel):
    source_dir: Path
    processed_dir: Path
    splits_dir: Path
    model_assets_dir: Path


class PipelineConfig(BaseModel):
    min_user_ratings: int = 10
    min_movie_ratings: int = 20
    train_ratio: float = 0.8
    val_ratio: float = 0.1


class ALSConfig(BaseModel):
    factors: int = 64
    regularization: float = 0.1
    iterations: int = 15
    alpha: int = 15


class FMConfig(BaseModel):
    no_components: int = 32
    epochs: int = 15
    num_threads: int = 4


class SVMConfig(BaseModel):
    c: float = 1.0
    max_iter: int = 2000
    negative_sampling_ratio: float = 3.0
    random_state: int = 42
    use_metadata_features: bool = True
    release_year_bucket_size: int = 10


class ModelsConfig(BaseModel):
    als: ALSConfig = ALSConfig()
    fm: FMConfig = FMConfig()
    svm: SVMConfig = SVMConfig()


class Config(BaseModel):
    data_dirs: DataConfig
    pipeline: PipelineConfig = PipelineConfig()
    models: ModelsConfig = ModelsConfig()


def load_config(path: Path = _DEFAULT_CONFIG) -> Config:
    """Load and validate pipeline config from YAML. Resolves relative paths against the recommender root."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    data = raw.get("data_dirs", {})
    for key in ("source_dir", "processed_dir", "splits_dir", "model_assets_dir"):
        val = data.get(key)
        if val and not Path(val).is_absolute():
            data[key] = str(_RECOMMENDER_ROOT / val)

    config = Config(**raw)

    for dir_path in (
        config.data_dirs.processed_dir,
        config.data_dirs.splits_dir,
        config.data_dirs.model_assets_dir,
    ):
        dir_path.mkdir(parents=True, exist_ok=True)

    return config
