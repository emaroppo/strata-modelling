"""The model catalog's schema: what was trained, from what, and how it did.

Runs are a chain, metrics are rows, and predictions live beside the runs
that produced them. See ``docs/adr/0005`` and ``docs/adr/0006``.
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
    # A timestamp and a host token, minted where the run happened; unique
    # across stores without coordination. See docs/adr/0005.
    Column("id", String(40), primary_key=True),
    # What this run continued from. Null for a cold start, which is the only
    # run whose numbers stand entirely on their own.
    Column("parent_run_id", ForeignKey("run.id"), nullable=True),
    # Which machine trained it. Provenance is a column rather than part of
    # the id: an id is immutable and a machine can be renamed or handed on.
    Column("origin", String(64), nullable=True),
    # Which catalog the dataset belongs to; a dataset name means something
    # within one. Null is unknown, not "some other". See docs/adr/0008.
    Column("catalog_id", String(64), nullable=True),
    # Lineage back into the catalog: these three resolve to the exact
    # samples and annotations behind the checkpoint.
    Column("dataset", String(255), nullable=False),
    # Null when no dataset version describes what the run trained on.
    Column("dataset_version", Integer, nullable=True),
    Column("label_set", String(255), nullable=False),
    Column("model", String(255), nullable=False),
    # Bumped when a change makes old checkpoints unreadable. Recorded per
    # run because the model can be upgraded underneath a project, and a
    # checkpoint whose neurons no longer match its class list fails silently.
    Column("model_version", String(32), nullable=False),
    Column("params", JSON, nullable=True),
    # The class list *as trained*, by position: what the checkpoint encodes.
    # See docs/adr/0005.
    Column("classes", JSON, nullable=False),
    Column("checkpoint", Text, nullable=True),
    # Which experiment file asked for this run, by the file's hash, so a
    # study is a query here rather than a walk of the ledger. Null for a run
    # nothing orchestrated. The trial is in the ledger, not on the run.
    Column("experiment_id", String(64), nullable=True),
    Column("created_at", DateTime, server_default=func.now()),
    Index("ix_run_dataset", "dataset", "dataset_version"),
    Index("ix_run_experiment", "experiment_id"),
)


#: What a run saw: which side each sample of its manifest was on. The
#: realisation of the split, per run, whether the sides were inherited from
#: the catalog's version or drawn from a seed — the two read the same here,
#: and a surprising number can be asked which samples the model was shown.
run_sample = Table(
    "run_sample",
    metadata,
    Column("run_id", ForeignKey("run.id", ondelete="CASCADE"), primary_key=True),
    Column("checksum", String(64), primary_key=True),
    Column("side", String(8), nullable=False),
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


#: What a run said about a sample, kept so it need not be asked twice.
#: Nothing here is ever invalidated: the key names every input. See
#: docs/adr/0006.
prediction = Table(
    "prediction",
    metadata,
    Column("run_id", String(40), primary_key=True),
    Column("checksum", String(64), primary_key=True),
    # The third input: what the model was told about the sample. Empty
    # where a project declares no features.
    Column("feature_digest", String(64), primary_key=True, server_default=""),
    # The value as the model produced it, stored whole rather than split
    # into columns: what a prediction looks like is the label schema's
    # business, and this only has to hand it back unchanged.
    Column("value", Text, nullable=False),
    Column("made_at", DateTime, server_default=func.now()),
)
