"""What crosses the wire between a caller and the modelling host, and the protocol both name."""

from pydantic import BaseModel, Field

from strata.contracts import AnyPrediction

from ..requests import GeneratedOutput, Run

#: What this release says over the wire; goes up only when an older side would misread a newer one
#: (``docs/adr/0007``).
PROTOCOL = 3


#: The header every request but ``/healthz`` names it in.
PROTOCOL_HEADER = "X-Strata-Protocol"


class ServiceError(Exception):
    """A request this host cannot honour."""


class SplitSides(BaseModel):
    """The split as the caller's directory realised it: one letter per sample in manifest order, and
    the order's digest.

    See ``docs/adr/0025``.
    """

    sides: str
    order_digest: str
    val_ratio: float | None = None
    holdout_ratio: float | None = None


class RoundRequest(BaseModel):
    """One training round, as asked for from somewhere else."""

    #: A dataset in the catalog this host serves, its version already frozen (``docs/adr/0003``).
    dataset_id: int
    #: What the caller means by that id, checked here by :func:`check_dataset` (``docs/adr/0008``).
    dataset_name: str
    dataset_version: int
    annotation_digest: str | None
    #: A registered short name; file references are refused by :func:`check_servable`.
    model: str
    params: dict = Field(default_factory=dict)
    #: Applied over ``params`` when the round starts cold; only this host knows whether it does
    #: (``docs/adr/0025``).
    fresh_params: dict = Field(default_factory=dict)
    #: Cold start, ignoring whatever this host last trained on this dataset.
    fresh: bool = False
    #: Which catalog the caller believes this host serves (``docs/adr/0008``).
    catalog_id: str
    #: The project's ``[[data.features]]`` declarations; the host resolves the values itself
    #: (``docs/adr/0011``).
    features: list[dict] = Field(default_factory=list)
    #: The experiment file asking for this round, by its hash; none from the command line
    #: (``docs/adr/0007``).
    experiment_id: str | None = None
    #: The sides as the caller's directory has them; None trains on the version's own
    #: (``docs/adr/0025``).
    split: SplitSides | None = None


class RoundResponse(BaseModel):
    run: Run
    metrics: dict[str, float] = Field(default_factory=dict)
    #: How many files were fetched, so a caller can tell a cache hit from a transfer.
    materialised: int = 0


class PredictionRequest(BaseModel):
    """Score some samples, by checksum, with a run this host holds. See ``docs/adr/0007``."""

    run_id: str
    #: Which of the run's checkpoints, by name; None is its last (``docs/adr/0054``).
    checkpoint: str | None = None
    checksums: list[str] = Field(default_factory=list)
    #: What the model is to be told about each sample, keyed by checksum (``docs/adr/0007``).
    features: dict[str, dict] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    #: Checksum to the value the model produced; an unknown sample is absent (``docs/adr/0007``).
    predictions: dict[str, AnyPrediction] = Field(default_factory=dict)
    #: Asked about but not in this catalog.
    unknown: list[str] = Field(default_factory=list)


class GenerationInputRef(BaseModel):
    """One input as it travels: values, and samples by checksum. See ``docs/adr/0007`` and
    ``docs/adr/0054``.
    """

    values: dict = Field(default_factory=dict)
    #: Checksums by the role each sample fills, in order within a role.
    samples: dict[str, list[str]] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    """Generate from a media model's run this host holds; positional, as
    :class:`strata.modelling.requests.GenerateRequest` is.
    """

    run_id: str
    checkpoint: str | None = None
    inputs: list[GenerationInputRef] = Field(default_factory=list)
    settings: list[dict] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)


class GenerationResponse(BaseModel):
    #: One per input, in order. Nothing is written on this host.
    outputs: list[GeneratedOutput] = Field(default_factory=list)
