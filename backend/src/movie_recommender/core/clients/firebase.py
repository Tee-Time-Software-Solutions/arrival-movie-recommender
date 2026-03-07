import logging
import firebase_admin
from firebase_admin import credentials
from movie_recommender.core.settings import AppSettings

logger = logging.getLogger(__name__)


def initialize_firebase(settings: AppSettings):
    logger.debug("Initializing firebase...")

    fb = settings.firebase

    cert_dict = {
        "type": "service_account",
        "project_id": fb.firebase_project_id,
        "private_key_id": fb.firebase_private_key_id,
        "private_key": fb.firebase_private_key,
        "client_email": fb.firebase_client_email,
        "client_id": fb.firebase_client_id,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    cred = credentials.Certificate(cert_dict)

    try:
        firebase_admin.get_app()
        logger.debug("Firebase app already initialized")
    except ValueError:
        firebase_admin.initialize_app(cred)
        logger.info("Firebase app initialized successfully")
