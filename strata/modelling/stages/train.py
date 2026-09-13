"""The train stage: a run from a materialised directory, here or on the modelling host."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from strata.labels import order_digest, sides_string

from .. import handlers
from ..remote.checks import check_catalog
from ..remote.client import RemoteError
from ..remote.wire import RoundRequest, SplitSides
from ..requests import Run, TrainRequest
from ._context import Context, DatasetIdentity, StageError, Strict, _manifest


class TrainStageRequest(Strict):
    """One round of training, wherever it runs.

    Locally the directory is what is trained from. Remotely the host
    materialises for itself from ``dataset``, and the directory is not
    needed — a host has the bucket; shipping it the files would be paying
    the network to avoid using the network.
    """

    dataset_dir: Path | None = None
    dataset: DatasetIdentity | None = None
    model: str
    params: dict = Field(default_factory=dict)
    #: Applied over ``params`` when the round starts cold.
    fresh_params: dict = Field(default_factory=dict)
    #: The warm-start policy, written down: cold, or from a named parent,
    #: or — neither given — from the newest run over the dataset.
    fresh: bool = False
    parent: str | None = None
    #: Declarations only, for the host to resolve as it materialises.
    features: list[dict] = Field(default_factory=list)
    #: The experiment file asking, by its hash, recorded on the run wherever
    #: it is made. None from the command line.
    experiment_id: str | None = None


class TrainRecord(Strict):
    run_id: str
    parent_run_id: str | None
    where: Literal["local", "remote"]
    dataset: str
    dataset_version: int | None
    model: str
    model_version: str
    classes: list[str]
    checkpoint: Path | None
    metrics: dict[str, float]
    #: Samples the host fetched to build its copy; zero locally and on a
    #: cache hit there.
    materialised: int = 0


def train(request: TrainStageRequest, context: Context) -> TrainRecord:
    if context.host is None:
        return _train_here(request, context)
    return _train_there(request, context)


def _train_here(request: TrainStageRequest, context: Context) -> TrainRecord:
    if request.dataset_dir is None:
        raise StageError("Training here needs a materialised directory; none was given.")
    manifest = _manifest(request.dataset_dir)
    store = context.store

    if request.parent is not None:
        parent = store.get(request.parent)
        if parent is None:
            raise StageError(f"No run {request.parent!r} to continue from.")
    elif request.fresh:
        parent = None
    else:
        # Not past a re-split: a run from before it may have trained on
        # what this version holds out
        parent = store.latest(
            manifest.dataset, manifest.catalog_id, since_version=manifest.sides_from_version
        )
    # Asked after the parent is known, not before. Fresh is a request and
    # cold is an outcome; they part company when nothing has trained on this
    # dataset yet, and a cold run then trained for as long as a warm one.
    cold = parent is None
    params = {**request.params, **request.fresh_params} if cold else dict(request.params)

    run = handlers.train(
        TrainRequest(
            dataset_dir=request.dataset_dir,
            model=request.model,
            params=params,
            parent_run_id=parent.id if parent else None,
            experiment_id=request.experiment_id,
        ),
        store,
        on_epoch=context.on_epoch,
    )
    return _record(run, "local")


def _train_there(request: TrainStageRequest, context: Context) -> TrainRecord:
    if request.dataset is None:
        raise StageError(
            "Training on the modelling host needs the dataset's identity — id, name, "
            "version, digest and catalog — for the host to check it means the same there."
        )
    host = context.host
    client = context.client(host.url, host.token)
    # Asked before anything is sent. A host on another catalog would refuse
    # the round anyway; asking first says which machine to repoint.
    served = client.served_catalog()
    check_catalog(request.dataset.catalog_id, served.get("id"))
    # The split as this side's directory has it, inherited or drawn, sent
    # positionally with a proof of the order. Without it the host would
    # train on the version's own sides, and a drawn holdout would be scored
    # on samples the host had trained on.
    split = None
    if request.dataset_dir is not None:
        held = _manifest(request.dataset_dir)
        split = SplitSides(
            sides=sides_string(held),
            order_digest=order_digest(held),
            val_ratio=held.val_ratio,
            holdout_ratio=held.holdout_ratio,
        )
    job = client.submit(
        RoundRequest(
            dataset_id=request.dataset.dataset_id,
            dataset_name=request.dataset.name,
            dataset_version=request.dataset.version,
            annotation_digest=request.dataset.annotation_digest,
            catalog_id=request.dataset.catalog_id,
            model=request.model,
            params=request.params,
            fresh_params=request.fresh_params,
            fresh=request.fresh,
            features=request.features,
            experiment_id=request.experiment_id,
            split=split,
        )
    )
    finished = client.follow(job["id"], on_state=context.on_state)
    if finished.get("state") != "done":
        raise RemoteError(f"The host reports job {job['id']} failed: {finished.get('error')}")
    result = finished["result"]
    record = _record(Run.model_validate(result["run"]), "remote")
    return record.model_copy(
        update={
            "metrics": dict(result.get("metrics", {})),
            "materialised": result.get("materialised", 0),
        }
    )


def _record(run: Run, where: str) -> TrainRecord:
    return TrainRecord(
        run_id=run.id,
        parent_run_id=run.parent_run_id,
        where=where,
        dataset=run.dataset,
        dataset_version=run.dataset_version,
        model=run.model,
        model_version=run.model_version,
        classes=list(run.classes),
        checkpoint=run.checkpoint,
        metrics=dict(run.metrics),
    )
