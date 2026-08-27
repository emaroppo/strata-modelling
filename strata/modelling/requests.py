"""The request is the contract.

These types are the whole interface between a caller and training, whether
the call is in-process or over a wire. Defining them first is what stops the
wire format being retrofitted onto an interface that grew in-process — and
what lets the HTTP adapter, when it arrives, add transport and nothing else.

Nothing here names a catalog or a database. A train request points at a
materialised directory, which is self-contained by design.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from strata.labels import AnyPrediction


class TrainRequest(BaseModel):
    """Everything needed to run one training job."""

    #: A materialised dataset: a manifest and its files. Self-contained, so
    #: the trainer needs no catalog and no database.
    dataset_dir: Path
    #: A registered short name, or a ``module:Class`` / ``file.py:Class``
    #: reference. The backend decides whether it can serve it — a capability
    #: list fetched earlier would be stale by now.
    model: str
    params: dict = Field(default_factory=dict)
    #: Continue from this run's checkpoint. The default, warm-starting from
    #: the newest run over the same dataset, is the caller's policy to apply
    #: rather than a default hidden in here.
    parent_run_id: str | None = None


class PredictRequest(BaseModel):
    """Everything needed to run inference."""

    #: Which run's checkpoint to use.
    run_id: str
    #: Absolute paths. Predicting over samples that are not in any dataset
    #: is the normal case — that is what the unlabelled pool is.
    paths: list[Path] = Field(default_factory=list)


class ScoredPath(BaseModel):
    """One model output, tied to the file it was made from.

    Not a prediction — it *holds* one. The distinction earns its keep: when
    both were called Prediction, code unwrapped .value in some places and
    not others, and a cache ended up storing wrappers that read back as
    empty values.
    """

    path: Path
    #: Whatever the model's task emits. Naming one concrete type here
    #: refused every span and box prediction on the way out of a scoring
    #: pass, which is the last place the wrapper travels before the
    #: review queue is ranked.
    value: AnyPrediction


class Run(BaseModel):
    """A training job that happened."""

    id: str
    parent_run_id: str | None = None
    #: Which machine trained this, for when two stores are merged.
    origin: str | None = None
    #: Which catalog the dataset belongs to. Null for a run recorded before
    #: catalogs had identities.
    catalog_id: str | None = None

    @property
    def short(self) -> str:
        """The id without its microseconds, for showing a person.

        Full ids are what everything keys on and what a merge needs; a
        report full of thirty-character strings is unreadable, and the
        microseconds are the part nobody is reading.
        """
        stamp, _, host = self.id.partition("-")
        return f"{stamp[:15]}-{host}" if host else self.id
    dataset: str
    #: None when nothing materialised describes this run's data.
    dataset_version: int | None = None
    label_set: str
    model: str
    model_version: str
    params: dict = Field(default_factory=dict)
    #: As trained, by position. The label set may have gained classes since.
    classes: list[str] = Field(default_factory=list)
    checkpoint: Path | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
