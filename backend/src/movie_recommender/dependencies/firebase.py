from typing import Callable, Dict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from firebase_admin import auth

oauth2scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_user(
    email_needs_verification: bool = False, user_private_route: bool = False
) -> Callable:
    def get_token_dependency(
        request: Request, token: str = Depends(oauth2scheme)
    ) -> Dict[str, any]:
        try:
            decoded_token = auth.verify_id_token(token)
            user_record = auth.get_user(decoded_token["uid"])
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
            )

        if email_needs_verification and not user_record.email_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email not verified",
            )

        if user_private_route:
            user_id_path = request.path_params.get("user_id")
            if not user_id_path:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Route is marked as private but has no {user_id} path param",
                )
            if decoded_token["uid"] != user_id_path:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: you can only access your own resources",
                )

        return {
            **decoded_token,
            "email_verified": user_record.email_verified,
            "display_name": user_record.display_name,
        }

    return get_token_dependency
