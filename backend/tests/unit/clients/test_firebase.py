"""
Unit tests for initialize_firebase.

Mocks the firebase_admin SDK entirely — no credentials or network needed.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from movie_recommender.core.clients.firebase import initialize_firebase


def _make_settings(
    project_id: str = "test-project",
    private_key_id: str = "key-id-123",
    private_key: str = "test-private-key-value",
    client_email: str = "svc@test.iam.gserviceaccount.com",
    client_id: str = "99999",
) -> MagicMock:
    settings = MagicMock()
    settings.firebase.firebase_project_id = project_id
    settings.firebase.firebase_private_key_id = private_key_id
    settings.firebase.firebase_private_key = private_key
    settings.firebase.firebase_client_email = client_email
    settings.firebase.firebase_client_id = client_id
    return settings


@patch("movie_recommender.core.clients.firebase.firebase_admin.initialize_app")
@patch("movie_recommender.core.clients.firebase.firebase_admin.get_app")
@patch("movie_recommender.core.clients.firebase.credentials.Certificate")
class TestInitializeFirebase:
    def test_initializes_app_on_first_call(
        self, mock_cert, mock_get_app, mock_init_app
    ):
        mock_get_app.side_effect = ValueError("no app")

        initialize_firebase(_make_settings())

        mock_init_app.assert_called_once()

    def test_skips_init_when_app_already_exists(
        self, mock_cert, mock_get_app, mock_init_app
    ):
        mock_get_app.return_value = MagicMock()

        initialize_firebase(_make_settings())

        mock_init_app.assert_not_called()

    def test_certificate_is_always_built(self, mock_cert, mock_get_app, mock_init_app):
        """Certificate is constructed regardless of whether the app is new."""
        mock_get_app.return_value = MagicMock()

        initialize_firebase(_make_settings())

        mock_cert.assert_called_once()

    def test_cert_dict_has_correct_structure(
        self, mock_cert, mock_get_app, mock_init_app
    ):
        mock_get_app.side_effect = ValueError()
        settings = _make_settings()

        initialize_firebase(settings)

        cert_dict = mock_cert.call_args[0][0]
        assert cert_dict["type"] == "service_account"
        assert cert_dict["project_id"] == settings.firebase.firebase_project_id
        assert cert_dict["private_key_id"] == settings.firebase.firebase_private_key_id
        assert cert_dict["private_key"] == settings.firebase.firebase_private_key
        assert cert_dict["client_email"] == settings.firebase.firebase_client_email
        assert cert_dict["client_id"] == settings.firebase.firebase_client_id
        assert "auth_uri" in cert_dict
        assert "token_uri" in cert_dict

    def test_cert_is_passed_to_initialize_app(
        self, mock_cert, mock_get_app, mock_init_app
    ):
        mock_get_app.side_effect = ValueError()
        fake_cred = MagicMock()
        mock_cert.return_value = fake_cred

        initialize_firebase(_make_settings())

        mock_init_app.assert_called_once_with(fake_cred)
