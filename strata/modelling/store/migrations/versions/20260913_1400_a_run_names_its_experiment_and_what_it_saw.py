"""a run names its experiment and what it saw

Revision ID: 9b1e4c7d2a53
Revises: 4d54c92d74f2

Two additions, both nullable or empty for every run already recorded. A
run gains the id of the experiment file that asked for it; a run recorded
by hand has none. And ``run_sample`` records which side each sample of
the run's manifest was on. Existing runs have no rows there, and a missing
row means unknown, not empty. See ``docs/adr/0005``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9b1e4c7d2a53"
down_revision: str | None = "4d54c92d74f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("run") as batch:
        batch.add_column(sa.Column("experiment_id", sa.String(64), nullable=True))
        batch.create_index("ix_run_experiment", ["experiment_id"])
    op.create_table(
        "run_sample",
        sa.Column(
            "run_id", sa.String(40), sa.ForeignKey("run.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("checksum", sa.String(64), primary_key=True),
        sa.Column("side", sa.String(8), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("run_sample")
    with op.batch_alter_table("run") as batch:
        batch.drop_index("ix_run_experiment")
        batch.drop_column("experiment_id")
