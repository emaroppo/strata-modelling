"""a run keeps every checkpoint

Revision ID: 7b02a738240e
Revises: c7a2e91f5b04

A ``checkpoint`` table, one row per checkpoint a run kept. Every run
already recorded with a checkpoint gets one row, named ``final``, from its
``checkpoint`` column, hashed now where the file is there to hash. The
column stays, as the run's last.

The prediction cache's key gains the checkpoint, by name. Existing rows
get their run's one checkpoint, ``final``, or the empty name for a run
that kept none. Rebuilt rather than altered because SQLite cannot add a
column to a primary key. See ``docs/adr/0054``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from alembic import op

revision: str = "7b02a738240e"
down_revision: str | None = "c7a2e91f5b04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


#: What a run that never named a checkpoint calls its one.
FINAL = "final"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(path: Path) -> str | None:
    """As ``checkpoint_digest`` in the store, copied so this script never changes under it."""
    if path.is_file():
        return _file_digest(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for name in sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()):
        digest.update(f"{name}\0{_file_digest(path / name)}\n".encode())
    return digest.hexdigest()


def upgrade() -> None:
    checkpoint = op.create_table(
        "checkpoint",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "run_id", sa.String(40), sa.ForeignKey("run.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("path", sa.Text(), nullable=True),
        sa.Column("digest", sa.String(64), nullable=True),
        sa.Column("saved_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("run_id", "name", name="uq_checkpoint_run_name"),
    )

    bind = op.get_bind()
    run = sa.table(
        "run",
        sa.column("id", sa.String),
        sa.column("checkpoint", sa.Text),
        sa.column("created_at", sa.DateTime),
    )
    rows = bind.execute(
        sa.select(run.c.id, run.c.checkpoint, run.c.created_at)
        .where(run.c.checkpoint.is_not(None))
        .order_by(run.c.created_at, run.c.id)
    ).all()
    if rows:
        bind.execute(
            sa.insert(checkpoint),
            [
                {
                    "run_id": run_id,
                    "name": FINAL,
                    "path": path,
                    "digest": _digest(Path(path)),
                    # When the run was recorded, which is when its one
                    # checkpoint was written
                    "saved_at": created_at or datetime.now(UTC).replace(tzinfo=None),
                }
                for run_id, path, created_at in rows
            ],
        )

    op.execute("ALTER TABLE prediction RENAME TO prediction_old")
    op.create_table(
        "prediction",
        sa.Column("run_id", sa.String(length=40), primary_key=True),
        sa.Column("checkpoint", sa.String(length=64), primary_key=True, server_default=""),
        sa.Column("checksum", sa.String(length=64), primary_key=True),
        sa.Column("feature_digest", sa.String(length=64), primary_key=True, server_default=""),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("made_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.execute(
        "INSERT INTO prediction (run_id, checkpoint, checksum, feature_digest, value, made_at) "
        "SELECT p.run_id, COALESCE(c.name, ''), p.checksum, p.feature_digest, p.value, p.made_at "
        "FROM prediction_old p LEFT JOIN checkpoint c ON c.run_id = p.run_id"
    )
    op.execute("DROP TABLE prediction_old")


def downgrade() -> None:
    """Back to a key without the checkpoint, and no table of them.

    Rows differing only by checkpoint collapse: the one kept is arbitrary,
    which is the honest consequence of a key that cannot tell them apart.
    The run's ``checkpoint`` column already names its last.
    """
    op.execute("ALTER TABLE prediction RENAME TO prediction_old")
    op.create_table(
        "prediction",
        sa.Column("run_id", sa.String(length=40), primary_key=True),
        sa.Column("checksum", sa.String(length=64), primary_key=True),
        sa.Column("feature_digest", sa.String(length=64), primary_key=True, server_default=""),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("made_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.execute(
        "INSERT OR IGNORE INTO prediction (run_id, checksum, feature_digest, value, made_at) "
        "SELECT run_id, checksum, feature_digest, value, made_at FROM prediction_old"
    )
    op.execute("DROP TABLE prediction_old")
    op.drop_table("checkpoint")
