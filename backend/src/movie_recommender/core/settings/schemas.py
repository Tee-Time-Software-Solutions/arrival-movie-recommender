from typing import Literal

from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class AppLogicSettings(BaseModel):
    batch_size: int
    queue_min_capacity: int
    learning_rate: float = 0.05
    norm_cap: float = 10.0
    over_fetch_factor: int = 2


class TMDBSettings(BaseModel):
    api_key: str
    img_url: str
    base_url: str


class RedisSettings(BaseModel):
    """Redis connection settings."""

    url: str
    max_connections: int = 10


class DatabaseSettings(BaseModel):
    """MySQL database settings."""

    user: str
    password: str
    host: str
    port: str
    database: str
    sync_driver: str
    async_driver: str


class StorageSettings(BaseModel):
    """Cloud storage settings (AWS S3 or Azure)."""

    provider: Literal["aws", "azure"]
    s3_bucket_name: str | None = None
    aws_region: str | None = None
    azure_container_name: str | None = None
    azure_account_name: str | None = None
    azure_account_key: str | None = None


class Neo4jSettings(BaseModel):
    uri: str
    username: str
    password: str
    database: str = "neo4j"


class OpenRouterSettings(BaseModel):
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "google/gemini-2.5-flash"


class FirebaseSettings(BaseModel):
    firebase_project_id: str
    firebase_private_key_id: str
    firebase_private_key: str
    firebase_client_email: str
    firebase_client_id: str
