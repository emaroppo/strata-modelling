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

Predictions live here too, beside the runs that produced them. A run id
means something only within one store, so a cache keyed on one belongs in
the same database — and the machine that made a prediction is the one that
should keep it, because the answer is the same for every caller.
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
    # A timestamp and a random suffix, minted where the run happened. An
    # autoincrementing integer means something only inside one store, and
    # there are two: a project keeps runs beside its own checkpoints and a
    # modelling host keeps its own. Both numbered from one, so the same
    # number named different models and nothing said so.
    Column("id", String(40), primary_key=True),
    # What this run continued from. Null for a cold start, which is the only
    # run whose numbers stand entirely on their own.
    Column("parent_run_id", ForeignKey("run.id"), nullable=True),
    # Which machine trained it. Provenance is a column rather than part of
    # the id: an id is immutable and a machine can be renamed or handed on.
    Column("origin", String(64), nullable=True),
    # Which catalog the dataset belongs to. A dataset name means something
    # within one, and a host can serve more than one — so without this,
    # "the latest run over demo" is a question with two answers.
    Column("catalog_id", String(64), nullable=True),
    # Lineage back into the catalog: these three resolve to the exact
    # samples and annotations behind the checkpoint.
    Column("dataset", String(255), nullable=False),
    # Null when no dataset version describes what the run trained on, which
    # is the state of anything imported from before the catalog existed. A
    # round number is not a dataset version, and standing one in for the
    # other made an imported round read as though it shared data with a
    # catalog one that happened to carry the same number.
    Column("dataset_version", Integer, nullable=True),
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


#: What a run said about a sample, kept so it need not be asked twice.
#:
#: Ranking a review queue needs a score for every unlabelled sample, not
#: just the ones about to be shown, so scoring a pool is minutes of GPU
#: whether it surfaces 200 samples or 20. Nothing here is ever invalidated,
#: and that is a property rather than an omission: a prediction is a
#: function of a checkpoint and some bytes, and both are immutable.
#:
#: Keyed on the checksum rather than a sample id, because an id belongs to
#: one catalog's numbering while the bytes are what the model actually saw.
prediction = Table(
    "prediction",
    metadata,
    Column("run_id", String(40), primary_key=True),
    Column("checksum", String(64), primary_key=True),
    # The third input. A prediction is a function of a checkpoint, some
    # bytes *and whatever the model was told about the sample* — and unlike
    # the first two, the third can change: a feature is another label set's
    # answer, under review by whoever owns it, so a correction is the
    # ordinary case rather than the exception.
    #
    # In the key rather than a policy to invalidate on, which is what lets
    # the note above stay literally true. Nothing here is ever invalidated;
    # the key simply names every input now, so a corrected feature is a
    # miss and the old row remains the right answer for the inputs it was
    # computed from. Empty where a project declares no features, which is
    # what every row written before this column was.
    Column("feature_digest", String(64), primary_key=True, server_default=""),
    # The value as the model produced it, stored whole rather than split
    # into columns: what a prediction looks like is the label schema's
    # business, and this only has to hand it back unchanged.
    Column("value", Text, nullable=False),
    Column("made_at", DateTime, server_default=func.now()),
)
