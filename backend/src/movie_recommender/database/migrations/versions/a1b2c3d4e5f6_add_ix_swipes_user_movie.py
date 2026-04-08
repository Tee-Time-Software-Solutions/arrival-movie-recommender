"""add ix_swipes_user_movie composite index

Revision ID: a1b2c3d4e5f6
Revises: 878c2868c8fe
Create Date: 2026-04-03 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "1b39ff3417c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_swipes_user_movie", "swipes", ["user_id", "movie_id"])


def downgrade() -> None:
    op.drop_index("ix_swipes_user_movie", table_name="swipes")
