"""
Unit tests for the verify_user dependency.

All Firebase SDK calls are mocked — no credentials or network needed.
The inner closure (get_token_dependency) is called directly, bypassing
FastAPI's Depends machinery, which only resolves at request time.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from movie_recommender.dependencies.firebase import verify_user

FAKE_UID = "test-uid-123"
FAKE_TOKEN = "fake.firebase.id.token"
DECODED_TOKEN = {
    "uid": FAKE_UID,
    "email": "test@example.com",
    "iat": 1_000_000,
    "exp": 9_999_999,
}


def _make_request(user_id: str | None = None) -> MagicMock:
    request = MagicMock()
    request.path_params = {"user_id": user_id} if user_id else {}
    return request


def _make_user_record(email_verified: bool = True, display_name: str = "Test User") -> MagicMock:
    record = MagicMock()
    record.email_verified = email_verified
    record.display_name = display_name
    return record


@patch("movie_recommender.dependencies.firebase.auth.get_user")
@patch("movie_recommender.dependencies.firebase.auth.verify_id_token")
class TestVerifyUserHappyPath:
    def test_valid_token_returns_merged_user_data(self, mock_verify, mock_get_user):
        mock_verify.return_value = DECODED_TOKEN
        mock_get_user.return_value = _make_user_record()

        result = verify_user()(request=_make_request(), token=FAKE_TOKEN)

        assert result["uid"] == FAKE_UID
        assert result["email"] == "test@example.com"
        assert result["email_verified"] is True
        assert result["display_name"] == "Test User"

    def test_no_flags_ignores_email_verification(self, mock_verify, mock_get_user):
        mock_verify.return_value = DECODED_TOKEN
        mock_get_user.return_value = _make_user_record(email_verified=False)

        result = verify_user()(request=_make_request(), token=FAKE_TOKEN)

        assert result["uid"] == FAKE_UID

    def test_email_verified_passes_when_required(self, mock_verify, mock_get_user):
        mock_verify.return_value = DECODED_TOKEN
        mock_get_user.return_value = _make_user_record(email_verified=True)

        result = verify_user(email_needs_verification=True)(
            request=_make_request(), token=FAKE_TOKEN
        )

        assert result["uid"] == FAKE_UID

    def test_private_route_matching_uid_passes(self, mock_verify, mock_get_user):
        mock_verify.return_value = DECODED_TOKEN
        mock_get_user.return_value = _make_user_record()

        result = verify_user(user_private_route=True)(
            request=_make_request(user_id=FAKE_UID), token=FAKE_TOKEN
        )

        assert result["uid"] == FAKE_UID


@patch("movie_recommender.dependencies.firebase.auth.get_user")
@patch("movie_recommender.dependencies.firebase.auth.verify_id_token")
class TestVerifyUserFailureCases:
    def test_invalid_token_raises_401(self, mock_verify, mock_get_user):
        mock_verify.side_effect = Exception("Token expired or malformed")

        with pytest.raises(HTTPException) as exc:
            verify_user()(request=_make_request(), token=FAKE_TOKEN)

        assert exc.value.status_code == 401
        assert "Authentication failed" == exc.value.detail

    def test_unverified_email_raises_403_when_required(self, mock_verify, mock_get_user):
        mock_verify.return_value = DECODED_TOKEN
        mock_get_user.return_value = _make_user_record(email_verified=False)

        with pytest.raises(HTTPException) as exc:
            verify_user(email_needs_verification=True)(
                request=_make_request(), token=FAKE_TOKEN
            )

        assert exc.value.status_code == 403
        assert "Email not verified" in exc.value.detail

    def test_private_route_without_user_id_path_param_raises_500(
        self, mock_verify, mock_get_user
    ):
        mock_verify.return_value = DECODED_TOKEN
        mock_get_user.return_value = _make_user_record()

        with pytest.raises(HTTPException) as exc:
            verify_user(user_private_route=True)(
                request=_make_request(user_id=None), token=FAKE_TOKEN
            )

        assert exc.value.status_code == 500

    def test_private_route_mismatched_uid_raises_403(self, mock_verify, mock_get_user):
        mock_verify.return_value = DECODED_TOKEN
        mock_get_user.return_value = _make_user_record()

        with pytest.raises(HTTPException) as exc:
            verify_user(user_private_route=True)(
                request=_make_request(user_id="completely-different-uid"),
                token=FAKE_TOKEN,
            )

        assert exc.value.status_code == 403
        assert "Access denied" in exc.value.detail

    def test_http_exceptions_are_not_swallowed_by_outer_except(
        self, mock_verify, mock_get_user
    ):
        """
        HTTPExceptions raised inside verify_user (e.g. 403, 500) must not be
        re-caught and downgraded to a 401 by the outer try/except.
        """
        mock_verify.return_value = DECODED_TOKEN
        mock_get_user.return_value = _make_user_record(email_verified=False)

        with pytest.raises(HTTPException) as exc:
            verify_user(email_needs_verification=True)(
                request=_make_request(), token=FAKE_TOKEN
            )

        assert exc.value.status_code == 403  # must NOT be 401
