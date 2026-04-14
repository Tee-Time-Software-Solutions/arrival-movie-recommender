#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from movie_recommender.core.settings.main import AppSettings

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / "env_config/synced/.env.dev")


class Migrator:
    def __init__(self):
        self.db = AppSettings().database

    def start(self):
        self._create_database()
        self._run_migrations()

    def _create_database(self):
        print(f"Creating database '{self.db.database}' if not exists... Settings for db: {self.db}")
        try:
            conn = psycopg2.connect(
                host=self.db.host,
                port=int(self.db.port),
                user=self.db.user,
                password=self.db.password,
                database="postgres",
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.db.database}'")
                if not cur.fetchone():
                    cur.execute(f"CREATE DATABASE {self.db.database}")
                    print(f"✅ Database '{self.db.database}' created")
                else:
                    print(f"✅ Database '{self.db.database}' already exists")
            conn.close()
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            sys.exit(1)

    def _run_migrations(self):
        print("Running Alembic migrations...")
        try:
            print("Pending migrations:")
            subprocess.run(["alembic", "history", "--indicate-current"], check=True)
            subprocess.run(["alembic", "upgrade", "head"], check=True)
            print("✅ Migrations completed. Current revision:")
            subprocess.run(["alembic", "current"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Migration failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    Migrator().start()
