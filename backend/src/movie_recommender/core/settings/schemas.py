from typing import Literal

from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class AppLogicSettings(BaseModel):
    batch_size: int
    queue_min_capacity: int


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
