from typing import Dict

import logging

from movie_recommender.core.settings.main import AppSettings


# DONT CHANGE THIS SECTION

logger = logging.getLogger(__name__)


async def get_app_settings() -> Dict:
    app_settings = AppSettings()
    print(f"App settings: {app_settings}")  # Remove in prod
    return app_settings
