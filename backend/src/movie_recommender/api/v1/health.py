from typing import Dict

from fastapi import APIRouter, status

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health")


# Health
@router.get(path="/ping", status_code=status.HTTP_200_OK)
async def check_backend_health_endpoint() -> Dict:
    return {"response": "pong"}
