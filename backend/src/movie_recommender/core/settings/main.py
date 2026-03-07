import os
import logging
from typing import List

from movie_recommender.core.settings.schemas import (
    AppLogicSettings,
    FirebaSettings,
    RedisSettings,
    DatabaseSettings,
    StorageSettings,
    TMDBSettings,
)

logger = logging.getLogger(__name__)


def check_required(required: List):
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required database variables: {', '.join(missing)}")


class AppSettings:
    """Application settings - singleton pattern."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Load all settings from environment."""
        self.environment = self._get_environment()
        self.app_logic = self._load_app_logic_settings()
        self.tmdb = self._load_tmdb_settings()
        self.redis = self._load_redis_settings()
        self.firebase = self._load_firebase_settings()
        # self.database = self._load_database_settings() # TODO: implement db
        # self.storage = self._load_storage_settings() # TODO: implement storage

        logger.info(f"Settings initialized for environment: {self.environment}")

    def _get_environment(self) -> str:
        """Get and validate environment."""
        env = os.getenv("ENVIRONMENT", "dev").lower()
        valid_envs = ["test", "dev", "staging", "production"]
        if env not in valid_envs:
            raise ValueError(f"Invalid ENVIRONMENT: {env}. Must be one of {valid_envs}")
        return env

    def _load_app_logic_settings(self) -> AppLogicSettings:
        batch_size = int(os.getenv("BATCH_SIZE", "15"))
        queue_min_capacity = int(os.getenv("QUEUE_MIN_CAPACITY", "5"))
        return AppLogicSettings(
            batch_size=batch_size, queue_min_capacity=queue_min_capacity
        )

    def _load_tmdb_settings(self) -> TMDBSettings:
        return TMDBSettings(
            api_key=os.getenv("TMDB_API_KEY"),
            img_url=os.getenv("TMDB_IMG_URL"),
            base_url=os.getenv("TMDB_BASE_URL"),
        )

    def _load_redis_settings(self) -> RedisSettings:
        """Load Redis settings."""
        url = os.getenv("REDIS_URL")
        if not url:
            raise ValueError("REDIS_URL is required")

        return RedisSettings(
            url=url, max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
        )

    def _load_database_settings(self) -> DatabaseSettings:
        """Load database settings."""
        check_required(
            [
                "MYSQL_USER",
                "MYSQL_PASSWORD",
                "MYSQL_HOST",
                "MYSQL_PORT",
                "MYSQL_DATABASE",
                "MYSQL_SYNC_DRIVER",
                "MYSQL_ASYNC_DRIVER",
            ]
        )

        return DatabaseSettings(
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            host=os.getenv("MYSQL_HOST"),
            port=os.getenv("MYSQL_PORT"),
            database=os.getenv("MYSQL_DATABASE"),
            sync_driver=os.getenv("MYSQL_SYNC_DRIVER"),
            async_driver=os.getenv("MYSQL_ASYNC_DRIVER"),
        )

    def _load_storage_settings(self) -> StorageSettings:
        """Load cloud storage settings."""
        provider = os.getenv("CLOUD_PROVIDER", "aws").lower()

        if provider == "aws":
            s3_bucket = os.getenv("S3_MAIN_BUCKET_NAME")
            aws_region = os.getenv("AWS_MAIN_REGION")
            if not s3_bucket or not aws_region:
                raise ValueError(
                    "S3_MAIN_BUCKET_NAME and AWS_MAIN_REGION required for AWS"
                )

            return StorageSettings(
                provider="aws", s3_bucket_name=s3_bucket, aws_region=aws_region
            )

        elif provider == "azure":
            container = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
            account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
            account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
            if not all([container, account_name, account_key]):
                raise ValueError("Azure storage variables required for Azure")

            return StorageSettings(
                provider="azure",
                azure_container_name=container,
                azure_account_name=account_name,
                azure_account_key=account_key,
            )

        else:
            raise ValueError(
                f"Unsupported CLOUD_PROVIDER: {provider}. Use 'aws' or 'azure'"
            )

    def _load_firebase_settings(self) -> FirebaSettings:
        check_required(
            [
                "FIREBASE_PROJECT_ID",
                "FIREBASE_PRIVATE_KEY_ID",
                "FIREBASE_PRIVATE_KEY",
                "FIREBASE_CLIENT_EMAIL",
                "FIREBASE_CLIENT_ID",
            ]
        )

        return FirebaSettings(
            firebase_project_id=os.getenv("FIREBASE_PROJECT_ID"),
            firebase_private_key_id=os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            firebase_private_key=os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
            firebase_client_email=os.getenv("FIREBASE_CLIENT_EMAIL"),
            firebase_client_id=os.getenv("FIREBASE_CLIENT_ID"),
        )
