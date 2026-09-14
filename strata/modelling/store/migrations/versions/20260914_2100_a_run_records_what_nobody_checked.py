"""a run records what nobody checked

Revision ID: c7a2e91f5b04
Revises: 9b1e4c7d2a53

``run_sample`` gains, per sample the run saw, which import batch its
label arrived in and whether a person vouched for it, as the manifest
said. Both nullable: a run recorded before this has neither, and null
means unknown rather than unreviewed. With them a run can say how much
of what it learned from, per side and per batch, nobody checked.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7a2e91f5b04"
down_revision: str | None = "9b1e4c7d2a53"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("run_sample") as batch:
        batch.add_column(sa.Column("batch", sa.String(64), nullable=True))
        batch.add_column(sa.Column("reviewed", sa.Boolean(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("run_sample") as batch:
        batch.drop_column("reviewed")
        batch.drop_column("batch")
