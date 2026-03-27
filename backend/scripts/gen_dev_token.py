"""
Generates a Firebase ID token for a test user and writes it to .dev_token.

Usage:
    uv run scripts/gen_dev_token.py [--uid <uid>]

Requires FIREBASE_WEB_API_KEY in your .env.dev in addition to the
existing FIREBASE_* service account variables.

Find your Web API Key at:
    Firebase Console → Project Settings → General → Web API Key
"""

import argparse
import sys
from pathlib import Path
import httpx
from dotenv import load_dotenv

import os

import firebase_admin
from firebase_admin import auth, credentials

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / "env_config/synced/.env.dev")

DEV_TOKEN_PATH = ROOT / "./scripts/.dev_token"
EXCHANGE_URL = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
)


def _init_firebase() -> None:
    cert_dict = {
        "type": "service_account",
        "project_id": os.environ["FIREBASE_PROJECT_ID"],
        "private_key_id": os.environ["FIREBASE_PRIVATE_KEY_ID"],
        "private_key": os.environ["FIREBASE_PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": os.environ["FIREBASE_CLIENT_EMAIL"],
        "client_id": os.environ["FIREBASE_CLIENT_ID"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(credentials.Certificate(cert_dict))


def _exchange_for_id_token(custom_token: str, web_api_key: str) -> str:
    """Exchange a custom token for an ID token via Firebase REST API."""
    resp = httpx.post(
        EXCHANGE_URL,
        params={"key": web_api_key},
        json={"token": custom_token, "returnSecureToken": True},
    )
    resp.raise_for_status()
    return resp.json()["idToken"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uid",
        default="dev-test-user",
        help="Firebase UID for the test user (default: dev-test-user)",
    )
    args = parser.parse_args()

    web_api_key = os.getenv("FIREBASE_WEB_API_KEY")
    if not web_api_key:
        print("Error: FIREBASE_WEB_API_KEY is not set in .env.dev", file=sys.stderr)
        print(
            "Find it at: Firebase Console → Project Settings → General → Web API Key",
            file=sys.stderr,
        )
        sys.exit(1)

    _init_firebase()

    custom_token = auth.create_custom_token(args.uid).decode("utf-8")
    id_token = _exchange_for_id_token(custom_token, web_api_key)

    DEV_TOKEN_PATH.write_text(id_token)
    print(f"ID token written to {DEV_TOKEN_PATH}")
    print(f"\nUID: {args.uid}")
    print(f"\nToken (first 60 chars): {id_token[:60]}...")
    print("\nUse as: Authorization: Bearer <token>")


if __name__ == "__main__":
    main()
