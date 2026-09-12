"""a cached span carries labels

Revision ID: 4d54c92d74f2
Revises: 882788cac3cf

The prediction cache holds spans as the model produced them, and the model
wrote each with one ``label``. The catalog's spans were rewritten to
``labels``, a list, in its own chain; the cache follows, so a cached answer
keeps reading under the one form the value type accepts and nothing
computed is thrown away. The value column is text, so this is JSON in
Python. Re-running is a no-op.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "4d54c92d74f2"
down_revision: str | None = "882788cac3cf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_PREDICTION = sa.table(
    "prediction",
    sa.column("run_id", sa.String),
    sa.column("checksum", sa.String),
    sa.column("feature_digest", sa.String),
    sa.column("value", sa.Text),
)


def _to_labels(value: dict):
    if value.get("kind") != "spans":
        return None
    changed = False
    spans = []
    for span in value.get("values") or []:
        if "label" in span:
            span = dict(span)
            lone = span.pop("label")
            span.setdefault("labels", [lone] if lone else [])
            changed = True
        spans.append(span)
    return {**value, "values": spans} if changed else None


def _to_label(value: dict):
    if value.get("kind") != "spans":
        return None
    changed = False
    spans = []
    for span in value.get("values") or []:
        if "labels" in span:
            span = dict(span)
            labels = span.pop("labels")
            if len(labels) > 1:
                raise RuntimeError(
                    "A cached span carries more than one label, which the "
                    "single-label form cannot hold; this migration cannot be walked back."
                )
            span["label"] = labels[0] if labels else ""
            changed = True
        spans.append(span)
    return {**value, "values": spans} if changed else None


def _rewrite(convert) -> None:
    bind = op.get_bind()
    columns = _PREDICTION.c
    rows = bind.execute(
        sa.select(columns.run_id, columns.checksum, columns.feature_digest, columns.value)
        .where(columns.value.like('%"label"%'))
    ).all()
    updates = []
    for run_id, checksum, digest, raw in rows:
        new = convert(json.loads(raw))
        if new is not None:
            updates.append(
                {
                    "b_run": run_id,
                    "b_sum": checksum,
                    "b_digest": digest,
                    "new": json.dumps(new, separators=(",", ":")),
                }
            )
    if updates:
        bind.execute(
            sa.update(_PREDICTION)
            .where(columns.run_id == sa.bindparam("b_run"))
            .where(columns.checksum == sa.bindparam("b_sum"))
            .where(columns.feature_digest == sa.bindparam("b_digest"))
            .values(value=sa.bindparam("new")),
            updates,
        )


def upgrade() -> None:
    _rewrite(_to_labels)


def downgrade() -> None:
    _rewrite(_to_label)
