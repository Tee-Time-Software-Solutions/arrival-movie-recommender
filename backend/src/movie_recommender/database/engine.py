from movie_recommender.core.settings.main import AppSettings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


class DatabaseEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        db = AppSettings().database
        url = f"{db.async_driver}://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
        engine = create_async_engine(
            url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
