"""The model catalog's schema: what was trained, from what, and how it did.

Runs are a chain, metrics are rows, and predictions live beside the runs
that produced them. See ``docs/adr/0005`` and ``docs/adr/0006``.
"""

from sqlalchemy import (
    JSON,
    Boolean,
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
    UniqueConstraint,
    func,
)

metadata = MetaData()


run = Table(
    "run",
    metadata,
    # A timestamp and a host token, unique across stores without coordination. docs/adr/0005
    Column("id", String(40), primary_key=True),
    # Null for a cold start. docs/adr/0005
    Column("parent_run_id", ForeignKey("run.id"), nullable=True),
    # Which machine trained it. docs/adr/0005
    Column("origin", String(64), nullable=True),
    # Which catalog the dataset belongs to; null is unknown, not "some other". docs/adr/0008
    Column("catalog_id", String(64), nullable=True),
    # Lineage back into the catalog; the version is null when none describes the data.
    Column("dataset", String(255), nullable=False),
    Column("dataset_version", Integer, nullable=True),
    Column("label_set", String(255), nullable=False),
    Column("model", String(255), nullable=False),
    # docs/adr/0005
    Column("model_version", String(32), nullable=False),
    Column("params", JSON, nullable=True),
    # The class list as trained, by position. docs/adr/0005
    Column("classes", JSON, nullable=False),
    # The run's last checkpoint, denormalised so a reader need not join. docs/adr/0054
    Column("checkpoint", Text, nullable=True),
    # The experiment file that asked, by hash; the trial is in the ledger, not here. docs/adr/0005
    Column("experiment_id", String(64), nullable=True),
    Column("created_at", DateTime, server_default=func.now()),
    Index("ix_run_dataset", "dataset", "dataset_version"),
    Index("ix_run_experiment", "experiment_id"),
)


#: What a run saw: which side each sample of its manifest was on (docs/adr/0005).
run_sample = Table(
    "run_sample",
    metadata,
    Column("run_id", ForeignKey("run.id", ondelete="CASCADE"), primary_key=True),
    Column("checksum", String(64), primary_key=True),
    Column("side", String(8), nullable=False),
    #: The label's import batch and whether a person vouched for it; null is "not said", not False
    #: (docs/adr/0005).
    Column("batch", String(64), nullable=True),
    Column("reviewed", Boolean, nullable=True),
)


#: Every checkpoint a run kept, oldest first; ``final`` for a model that named none (docs/adr/0054).
checkpoint = Table(
    "checkpoint",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("run_id", ForeignKey("run.id", ondelete="CASCADE"), nullable=False),
    # As the model named it; unique within the run.
    Column("name", String(64), nullable=False),
    # Null where the file is not in this store. docs/adr/0005
    Column("path", Text, nullable=True),
    # By checkpoint_digest; null where the model wrote nothing.
    Column("digest", String(64), nullable=True),
    Column("saved_at", DateTime, nullable=False),
    UniqueConstraint("run_id", "name", name="uq_checkpoint_run_name"),
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


#: What a run said about a sample; the key names every input, so nothing is invalidated
#: (docs/adr/0006, docs/adr/0054).
prediction = Table(
    "prediction",
    metadata,
    Column("run_id", String(40), primary_key=True),
    # By name; empty for a run that kept none. docs/adr/0054
    Column("checkpoint", String(64), primary_key=True, server_default=""),
    Column("checksum", String(64), primary_key=True),
    # What the model was told about the sample; empty where no features are declared.
    Column("feature_digest", String(64), primary_key=True, server_default=""),
    # Stored whole. docs/adr/0006
    Column("value", Text, nullable=False),
    Column("made_at", DateTime, server_default=func.now()),
)
