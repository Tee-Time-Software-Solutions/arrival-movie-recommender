import os

from movie_recommender.core.settings.main import AppSettings

_TEST_ENV = {
    # Core
    "ENVIRONMENT": "test",
    "BATCH_SIZE": "15",
    "QUEUE_MIN_CAPACITY": "5",
    # Redis
    "REDIS_URL": "redis://localhost:6379/0",
    # TMDB
    "TMDB_API_KEY": "test-tmdb-key",
    "TMDB_IMG_URL": "https://image.tmdb.org/t/p/w500",
    "TMDB_BASE_URL": "https://api.themoviedb.org/3",
    # Firebase
    "FIREBASE_PROJECT_ID": "test-project",
    "FIREBASE_PRIVATE_KEY_ID": "test-key-id",
    "FIREBASE_PRIVATE_KEY": "test-private-key-value",
    "FIREBASE_CLIENT_EMAIL": "svc@test.iam.gserviceaccount.com",
    "FIREBASE_CLIENT_ID": "99999",
    # Database (unused but present for completeness)
    "MYSQL_HOST": "localhost",
    "MYSQL_PORT": "3306",
    "MYSQL_DATABASE": "test_db",
    "MYSQL_USER": "test_user",
    "MYSQL_PASSWORD": "test_password",
    "MYSQL_SYNC_DRIVER": "pymysql",
    "MYSQL_ASYNC_DRIVER": "aiomysql",
    # Storage
    "CLOUD_PROVIDER": "aws",
    "S3_MAIN_BUCKET_NAME": "test-bucket",
    "AWS_MAIN_REGION": "us-east-1",
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AZURE_STORAGE_ACCOUNT_NAME": "test-storage-acc",
    "AZURE_STORAGE_ACCOUNT_KEY": "test-storage-acc-key",
    "AZURE_STORAGE_CONTAINER_NAME": "test-images-container",
}


def pytest_configure(config):
    # Reset singleton so it picks up our env vars, not a stale instance
    # from a previous run (relevant in watch mode).
    AppSettings._instance = None

    for key, value in _TEST_ENV.items():
        os.environ[key] = value


def pytest_unconfigure(config):
    AppSettings._instance = None

    for key in _TEST_ENV:
        os.environ.pop(key, None)
