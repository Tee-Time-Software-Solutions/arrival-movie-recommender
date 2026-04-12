"""merge user_online_vectors and onboarding_completed heads

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7, b2c3d4e5f6g7
Create Date: 2026-04-12 13:00:00.000000

"""

from typing import Sequence, Union

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, tuple[str, ...]] = ("b2c3d4e5f6a7", "b2c3d4e5f6g7")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
