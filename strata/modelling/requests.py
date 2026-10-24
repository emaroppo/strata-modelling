"""The request is the contract.

These types are the whole interface between a caller and training, whether
the call is in-process or over a wire; the HTTP adapter adds transport and
nothing else. See ``docs/adr/0007``.

Nothing here names a catalog or a database. A train request points at a
materialised directory, which is self-contained by design.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata.contracts import AnyPrediction

from .model import GenerationInput


class TrainRequest(BaseModel):
    """Everything needed to run one training job."""

    #: A materialised dataset: a manifest and its files. Self-contained. See
    #: ``docs/adr/0004``.
    dataset_dir: Path
    #: A registered short name, or a ``module:Class`` / ``file.py:Class``
    #: reference. The backend decides whether it can serve it. See
    #: ``docs/adr/0034``.
    model: str
    params: dict = Field(default_factory=dict)
    #: Continue from this run's checkpoint. Choosing a parent is the
    #: caller's policy, not a default hidden in here. See ``docs/adr/0025``.
    parent_run_id: str | None = None
    #: Which of the parent's checkpoints, by name. None is its last. See
    #: ``docs/adr/0054``.
    parent_checkpoint: str | None = None
    #: Recorded on the run. The orchestrator sets it; a round from the
    #: command line has none. See ``docs/adr/0005``.
    experiment_id: str | None = None
    #: Train on this share of the training side, for a learning curve.
    #: Subsets are nested: for one ``subset_seed``, a smaller fraction's
    #: samples are always among a larger one's. Validation is never cut.
    train_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    subset_seed: int = 0


class PredictRequest(BaseModel):
    """Everything needed to run inference."""

    #: Which run's checkpoint to use.
    run_id: str
    #: Which of the run's checkpoints, by name. None is its last. See
    #: ``docs/adr/0054``.
    checkpoint: str | None = None
    #: Absolute paths. Predicting over samples that are not in any dataset
    #: is the normal case — that is what the unlabelled pool is.
    paths: list[Path] = Field(default_factory=list)
    #: What the model is told about each path, positional against ``paths``.
    #: Empty for a project that declares no features. Sent with the request
    #: rather than looked up by the handler. See ``docs/adr/0011``.
    features: list[dict] = Field(default_factory=list)


class ScoredPath(BaseModel):
    """One model output, tied to the file it was made from.

    Not a prediction — it *holds* one. See ``docs/adr/0006``.
    """

    path: Path
    #: Whatever the model's label type emits. See ``docs/adr/0004``.
    value: AnyPrediction


class GenerateRequest(BaseModel):
    """Everything needed to generate from a media model's run.

    Inputs, settings and seeds are positional: the nth output is made from
    the nth of each, so one request can sweep a setting. Outputs are
    returned, never written; the caller commits them. See ``docs/adr/0054``.
    """

    run_id: str
    #: Which of the run's checkpoints, by name. None is its last.
    checkpoint: str | None = None
    #: The values each output's prompt is filled from, and any samples it is
    #: given, as absolute paths by role.
    inputs: list[GenerationInput] = Field(default_factory=list)
    settings: list[dict] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _positional(self) -> "GenerateRequest":
        if not len(self.inputs) == len(self.settings) == len(self.seeds):
            raise ValueError(
                f"{len(self.inputs)} input(s), {len(self.settings)} setting(s) and "
                f"{len(self.seeds)} seed(s): they are positional, one of each per output."
            )
        return self


class GeneratedOutput(BaseModel):
    """One output, and everything that made it besides its input.

    Not yet a sample: the caller commits it as a generate run (record
    0053), knowing which input it was given. See ``docs/adr/0054``.
    """

    #: Base64 over a wire, so an image survives JSON as well as a text.
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    data: bytes
    #: Whether it stopped on its own rather than at a limit; None for a
    #: media with no such notion.
    ended: bool | None = None
    #: SHA-256 over the canonical JSON of the settings it was made with.
    settings_digest: str
    seed: int
    #: The checkpoint that made it, by name and by digest.
    checkpoint: str
    checkpoint_digest: str | None = None


class Checkpoint(BaseModel):
    """One checkpoint a run kept.

    ``final`` for a model that never named one. See ``docs/adr/0054``.
    """

    name: str
    #: None where the file is not in this store, as after a merge that left
    #: checkpoints behind. See ``docs/adr/0005``.
    path: Path | None = None
    #: SHA-256 over what the model wrote. See
    #: :func:`strata.modelling.store.runs.checkpoint_digest`.
    digest: str | None = None
    saved_at: datetime | None = None


class Run(BaseModel):
    """A training job that happened."""

    id: str
    parent_run_id: str | None = None
    #: Which machine trained this, for when two stores are merged.
    origin: str | None = None
    #: Which catalog the dataset belongs to. Null for a run recorded before
    #: catalogs had identities.
    catalog_id: str | None = None
    #: The experiment file that asked for this run, by its hash. None for a
    #: run nothing orchestrated.
    experiment_id: str | None = None

    @property
    def short(self) -> str:
        """The id without its microseconds, for showing a person.

        Full ids are what everything keys on. See ``docs/adr/0005``.
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
    #: The last of ``checkpoints``: what a reader that names none gets.
    checkpoint: Path | None = None
    #: Every checkpoint the run kept, oldest first. See ``docs/adr/0054``.
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    def checkpoint_named(self, name: str | None = None) -> Checkpoint | None:
        """One of the run's checkpoints by name, or its last when none is named."""
        if name is None:
            return self.checkpoints[-1] if self.checkpoints else None
        return next((c for c in self.checkpoints if c.name == name), None)
