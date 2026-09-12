"""a prediction names every input

Revision ID: 882788cac3cf
Revises: c6a165e0459f
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '882788cac3cf'
down_revision: str | None = 'c6a165e0459f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Widen the key, keeping every row already computed.

    Autogenerate offered `add_column` alone, which is not enough: the
    primary key is what stops two answers for one sample, and leaving it at
    (run_id, checksum) would make a re-score after a feature change collide
    with the row it is meant to supersede rather than sit beside it.

    Rebuilt rather than altered because the run store is SQLite, which
    cannot add a column to a primary key. Existing rows carry the empty
    digest, which is exactly what they are: predictions made when the model
    was told nothing.
    """
    op.execute("ALTER TABLE prediction RENAME TO prediction_old")
    op.create_table(
        "prediction",
        sa.Column("run_id", sa.String(length=40), primary_key=True),
        sa.Column("checksum", sa.String(length=64), primary_key=True),
        sa.Column(
            "feature_digest",
            sa.String(length=64),
            primary_key=True,
            server_default="",
        ),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("made_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.execute(
        "INSERT INTO prediction (run_id, checksum, feature_digest, value, made_at) "
        "SELECT run_id, checksum, '', value, made_at FROM prediction_old"
    )
    op.execute("DROP TABLE prediction_old")


def downgrade() -> None:
    """Back to two-thirds of a key.

    Rows differing only by digest collapse: the one kept is arbitrary,
    which is the honest consequence of a key that cannot tell them apart.
    """
    op.execute("ALTER TABLE prediction RENAME TO prediction_old")
    op.create_table(
        "prediction",
        sa.Column("run_id", sa.String(length=40), primary_key=True),
        sa.Column("checksum", sa.String(length=64), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("made_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.execute(
        "INSERT OR IGNORE INTO prediction (run_id, checksum, value, made_at) "
        "SELECT run_id, checksum, value, made_at FROM prediction_old"
    )
    op.execute("DROP TABLE prediction_old")
