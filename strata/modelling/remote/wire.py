"""What crosses the wire between a caller and the modelling host, and the protocol both name."""


from pydantic import BaseModel, Field

from strata.labels import AnyPrediction

from ..requests import Run

#: What this release says over the wire. Goes up only when an older side
#: would misread a newer one (``docs/adr/0007``).
PROTOCOL = 1


#: The header every request but ``/healthz`` names it in.
PROTOCOL_HEADER = "X-Strata-Protocol"


class ServiceError(Exception):
    """A request this host cannot honour."""


class RoundRequest(BaseModel):
    """One training round, as asked for from somewhere else."""

    #: A dataset in the catalog this host is configured against. Its version
    #: is already frozen — the caller made it — so two hosts materialising
    #: the same id get the same bytes.
    dataset_id: int
    #: What the caller means by that id: the name, version and digest its
    #: own catalog gave. Checked against this host's catalog, because a copy
    #: of a catalog keeps the identity and numbers its datasets on its own.
    #: See :func:`check_dataset`.
    dataset_name: str
    dataset_version: int
    annotation_digest: str | None
    #: A registered short name. File references are refused: see
    #: :func:`check_servable`.
    model: str
    params: dict = Field(default_factory=dict)
    #: Applied over ``params`` when the round starts cold. Both sets travel
    #: because only this host knows whether it has a parent: a caller asking
    #: for a warm start against a store that turns out to be empty would
    #: otherwise get a cold run trained for as long as a warm one.
    fresh_params: dict = Field(default_factory=dict)
    #: Cold start, ignoring whatever this host last trained on this dataset.
    fresh: bool = False
    #: Which catalog the caller believes this host serves. Required: on
    #: another catalog the same id names different samples, and the round
    #: would succeed over the wrong data. See ``docs/adr/0008``.
    catalog_id: str
    #: What the model is to be told about each sample, as the project
    #: declares it under ``[[data.features]]``: ``{name, source, ref}``. The
    #: host resolves the values from its own catalog as it materialises, so
    #: only the declarations travel. Defaulted: a project declaring none
    #: sends none.
    features: list[dict] = Field(default_factory=list)


class RoundResponse(BaseModel):
    run: Run
    metrics: dict[str, float] = Field(default_factory=dict)
    #: How the dataset was obtained, so a caller can tell a cache hit from a
    #: transfer without reading the host's logs.
    materialised: int = 0


class PredictionRequest(BaseModel):
    """Score some samples with a run this host holds.

    Checksums rather than paths: a path names a file on the caller's
    machine, and the whole point is that the caller has none. This host has
    the bucket and a cache, so it can turn content into files itself.
    """

    run_id: str
    checksums: list[str] = Field(default_factory=list)
    #: What the model is to be told about each sample, by checksum. Keyed
    #: rather than positional because the response is keyed too, and a
    #: caller that has to keep two lists aligned across a wire eventually
    #: does not.
    features: dict[str, dict] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    #: Checksum to the value the model produced. Keyed rather than ordered,
    #: because a sample the catalog does not know is simply absent and a
    #: positional answer could not say which.
    predictions: dict[str, AnyPrediction] = Field(default_factory=dict)
    #: Asked about but not in this catalog.
    unknown: list[str] = Field(default_factory=list)
