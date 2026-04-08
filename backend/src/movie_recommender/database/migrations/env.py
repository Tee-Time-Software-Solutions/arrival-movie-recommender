from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from movie_recommender.core.settings.main import AppSettings
from movie_recommender.database.models import metadata
from sqlalchemy import create_engine, pool

# Load env vars for local runs; no-op in Docker where they're already set
load_dotenv(Path(__file__).parents[4] / "env_config/synced/.env.dev")

alembic_config = context.config

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

target_metadata = metadata


def _get_url() -> str:
    db = AppSettings().database
    return (
        f"{db.sync_driver}://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_get_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
