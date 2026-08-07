"""The model catalog's schema: what was trained, from what, and how it did.

Two things here are the reason this is a database rather than a folder of
JSON.

Runs are a **chain**. Each round warm-starts from the last, so a run has a
parent and its metrics only mean something relative to it — telling a real
improvement from a warm start's head start needs the edge, not just the
node.

Metrics are **rows**. "Accuracy per round" is the single most useful thing
to ask of a training history, and it should be a query rather than a glob
over directories and a parse.
"""

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
)

metadata = MetaData()


run = Table(
    "run",
    metadata,
    Column("id", Integer, primary_key=True),
    # What this run continued from. Null for a cold start, which is the only
    # run whose numbers stand entirely on their own.
    Column("parent_run_id", ForeignKey("run.id"), nullable=True),
    # Lineage back into the catalog: these three resolve to the exact
    # samples and annotations behind the checkpoint.
    Column("dataset", String(255), nullable=False),
    Column("dataset_version", Integer, nullable=False),
    Column("label_set", String(255), nullable=False),
    Column("model", String(255), nullable=False),
    # Bumped when a change makes old checkpoints unreadable. Recorded per
    # run because the model can be upgraded underneath a project, and a
    # checkpoint whose neurons no longer match its class list fails silently.
    Column("model_version", String(32), nullable=False),
    Column("params", JSON, nullable=True),
    # The class list *as trained*, by position. The label set can gain
    # classes afterwards; this is what the checkpoint actually encodes.
    Column("classes", JSON, nullable=False),
    Column("checkpoint", Text, nullable=True),
    Column("created_at", DateTime, server_default=func.now()),
    Index("ix_run_dataset", "dataset", "dataset_version"),
)


metric = Table(
    "metric",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("run_id", ForeignKey("run.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(64), nullable=False),
    Column("value", Float, nullable=False),
    # Null for a run's final figure; set for a point in its training curve.
    Column("epoch", Integer, nullable=True),
    Index("ix_metric_run", "run_id", "name"),
)
