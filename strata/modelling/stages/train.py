"""The train stage: a run from a materialised directory, here or on the modelling host."""

from pathlib import Path

from pydantic import Field

from strata.contracts import order_digest, sides_string

from .. import handlers
from ..remote.checks import check_catalog
from ..remote.client import RemoteError
from ..remote.wire import RoundRequest, SplitSides
from ..requests import Run, TrainRequest
from ._context import Context, DatasetIdentity, Host, Keyed, StageError, Strict, Where, _manifest


class TrainStageRequest(Keyed):
    """One round of training: from ``dataset_dir`` here, or from ``dataset`` on the host. See
    ``docs/adr/0007``.
    """

    later = frozenset({"parent_checkpoint"})

    dataset_dir: Path | None = None
    dataset: DatasetIdentity | None = None
    model: str
    params: dict = Field(default_factory=dict)
    #: Applied over ``params`` when the round starts cold.
    fresh_params: dict = Field(default_factory=dict)
    #: The warm-start policy: cold, from a named parent, or from the newest run over the dataset
    #: (``docs/adr/0025``).
    fresh: bool = False
    parent: str | None = None
    #: Which of ``parent``'s checkpoints to continue from; None is its last, and a parent must be
    #: named (``docs/adr/0054``).
    parent_checkpoint: str | None = None
    #: Declarations only, for the host to resolve as it materialises.
    features: list[dict] = Field(default_factory=list)
    #: The experiment file asking, by its hash; None from the command line.
    experiment_id: str | None = None
    #: A share of the training side, nested per ``subset_seed``; see
    #: :class:`strata.modelling.requests.TrainRequest`.
    train_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    subset_seed: int = 0


class TrainRecord(Strict):
    run_id: str
    parent_run_id: str | None
    where: Where
    dataset: str
    dataset_version: int | None
    model: str
    model_version: str
    classes: list[str]
    checkpoint: Path | None
    metrics: dict[str, float]
    #: Samples the host fetched to build its copy; zero locally and on a cache hit.
    materialised: int = 0


def train(request: TrainStageRequest, context: Context) -> TrainRecord:
    if context.host is None:
        return _train_here(request, context)
    return _train_there(request, context, context.host)


def _train_here(request: TrainStageRequest, context: Context) -> TrainRecord:
    if request.dataset_dir is None:
        raise StageError("Training here needs a materialised directory; none was given.")
    manifest = _manifest(request.dataset_dir)
    store = context.store
    if store is None:
        raise StageError("Training here needs a run store; none was given.")

    if request.parent_checkpoint is not None and request.parent is None:
        raise StageError(
            f"parent_checkpoint {request.parent_checkpoint!r} names a checkpoint of no run: "
            f"name the run it belongs to as parent."
        )
    if request.parent is not None:
        parent = store.get(request.parent)
        if parent is None:
            raise StageError(f"No run {request.parent!r} to continue from.")
    elif request.fresh:
        parent = None
    else:
        # Not past a re-split. docs/adr/0024
        parent = store.latest(
            manifest.dataset, manifest.catalog_id, since_version=manifest.sides_from_version
        )
    # Fresh is a request, cold is an outcome. docs/adr/0025
    cold = parent is None
    params = {**request.params, **request.fresh_params} if cold else dict(request.params)

    run = handlers.train(
        TrainRequest(
            dataset_dir=request.dataset_dir,
            model=request.model,
            params=params,
            parent_run_id=parent.id if parent else None,
            parent_checkpoint=request.parent_checkpoint,
            experiment_id=request.experiment_id,
            train_fraction=request.train_fraction,
            subset_seed=request.subset_seed,
        ),
        store,
        on_epoch=context.on_epoch,
    )
    return _record(run, "local")


def _train_there(request: TrainStageRequest, context: Context, host: Host) -> TrainRecord:
    if request.train_fraction < 1.0:
        raise StageError(
            "A train_fraction below 1 is only supported for rounds trained here, "
            "not on a modelling host. Run the learning curve locally."
        )
    if request.parent_checkpoint is not None:
        raise StageError(
            "A parent_checkpoint is only supported for rounds trained here, not on a "
            "modelling host, which chooses the parent itself."
        )
    if request.dataset is None:
        raise StageError(
            "Training on the modelling host needs the dataset's identity — id, name, "
            "version, digest and catalog — for the host to check it means the same there."
        )
    client = context.client(host.url, host.token)
    # docs/adr/0008
    served = client.served_catalog()
    check_catalog(request.dataset.catalog_id, str(served.get("id") or ""))
    # The split as this side's directory has it, with a proof of the order. docs/adr/0025
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


def _record(run: Run, where: Where) -> TrainRecord:
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
