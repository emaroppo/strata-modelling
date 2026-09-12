"""Modelling's stages: training and scoring as an experiment asks for them.

``train`` takes a materialised directory and returns the run it recorded,
here or on the modelling host — the request says which by whether the
context names a host, and the record is the same shape either way.
``evaluate`` scores one side of a directory with a recorded run and
computes the number by one implementation, which is what makes two runs'
numbers comparable: a model reports whatever it likes about itself while
it trains, and that number is its own.

Requests and records are plain models. Nothing here names a catalog: a
directory is self-contained, and the remote branch sends a dataset's
identity for the host to resolve against its own.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from strata.common.stages import Stage
from strata.labels import MANIFEST_NAME, Manifest, feature_digest

from . import handlers
from .client import RemoteError, Trainer
from .predictions import PredictionCache
from .requests import PredictRequest, Run, TrainRequest
from .runs import RunStore
from .service import PredictionRequest, RoundRequest, check_catalog

DATASET_DIR = "dataset_dir"
RUN = "run"
METRICS = "metrics"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageError(Exception):
    """A request a stage cannot honour as it stands."""


@dataclass(frozen=True)
class Host:
    """A modelling host, as the settings name it."""

    url: str
    token: str


@dataclass
class Context:
    """The handles: this host's run store, and the other host if there is one."""

    store: RunStore
    host: Host | None = None
    #: What a model reports as it trains, forwarded; see :data:`EpochReport`.
    on_epoch: object = None
    #: A remote job's state as it is polled, for whoever is watching.
    on_state: object = None
    #: How to build the client, so a test can hand in a fake host.
    client: type = Trainer


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------


class DatasetIdentity(Strict):
    """What a dataset id means where the round was prepared, for a host to check."""

    dataset_id: int
    name: str
    version: int
    annotation_digest: str | None
    catalog_id: str


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
        parent = store.latest(manifest.dataset, manifest.catalog_id)
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


# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------


class EvaluateRequest(Strict):
    """Score one side of a directory with a run, by one implementation."""

    run_id: str
    dataset_dir: Path
    #: ``holdout`` is what a study reports on; ``val`` is what it selects on.
    side: Literal["val", "holdout"] = "holdout"


class ClassScore(Strict):
    precision: float
    recall: float
    f1: float
    #: How many held-out samples assert the class.
    support: int


class EvaluateRecord(Strict):
    run_id: str
    side: str
    where: Literal["local", "remote"]
    #: Samples with an answer on that side; the denominator.
    samples: int
    #: ``exact_match`` is the share of samples whose asserted set of classes
    #: is exactly the model's; ``precision``, ``recall`` and ``f1`` are micro,
    #: over every class assertion. A label set that carries two classes per
    #: sample scored by a model that asserts one has an exact match of zero
    #: and a precision worth reading, which is why both are here and neither
    #: is called accuracy.
    metrics: dict[str, float]
    per_class: dict[str, ClassScore] = Field(default_factory=dict)


def evaluate(request: EvaluateRequest, context: Context) -> EvaluateRecord:
    manifest = _manifest(request.dataset_dir)
    task = manifest.label_schema.task
    if task != "classification":
        # Honest rather than approximate: a set-of-classes score over spans
        # or boxes would be a number, and not the one anybody means by it.
        raise StageError(
            f"evaluate scores classification only for now, and this label set is "
            f"{task!r}. Span-level and box-level scoring are still to come."
        )
    samples = [s for s in manifest.samples if s.split == request.side and s.value is not None]
    if not samples:
        raise StageError(
            f"Nothing to score: {request.dataset_dir} has no answered sample on the "
            f"{request.side!r} side."
        )
    predictions, where = _predictions(request, context, samples)

    exact = 0
    counts: dict[str, list[int]] = {}  # class -> [tp, fp, fn]
    for sample in samples:
        truth = set(sample.value.values)
        guessed = set(predictions[sample.checksum].values)
        exact += truth == guessed
        for name in truth | guessed:
            tally = counts.setdefault(name, [0, 0, 0])
            tally[0] += name in truth and name in guessed
            tally[1] += name in guessed and name not in truth
            tally[2] += name in truth and name not in guessed
    tp = sum(t[0] for t in counts.values())
    fp = sum(t[1] for t in counts.values())
    fn = sum(t[2] for t in counts.values())
    precision, recall, f1 = _prf(tp, fp, fn)
    return EvaluateRecord(
        run_id=request.run_id,
        side=request.side,
        where=where,
        samples=len(samples),
        metrics={
            "exact_match": exact / len(samples),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        per_class={name: _class_score(tally) for name, tally in sorted(counts.items())},
    )


def _class_score(tally: list[int]) -> ClassScore:
    precision, recall, f1 = _prf(*tally)
    return ClassScore(precision=precision, recall=recall, f1=f1, support=tally[0] + tally[2])


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _predictions(request: EvaluateRequest, context: Context, samples) -> tuple[dict, str]:
    """What the run says about each sample, from the cache where it can be.

    A prediction is a function of a checkpoint, some bytes and the features
    the model was told, so the cache is keyed on all three and a scored
    sample is never scored twice.
    """
    digests = {s.checksum: feature_digest(s.features) for s in samples}
    checksums = [s.checksum for s in samples]

    if context.host is not None:
        client = context.client(context.host.url, context.host.token)
        job = client.predict(
            PredictionRequest(
                run_id=request.run_id,
                checksums=checksums,
                features={s.checksum: s.features for s in samples if s.features},
            )
        )
        finished = client.follow(job["id"], on_state=context.on_state)
        if finished.get("state") != "done":
            raise RemoteError(f"The host reports job {job['id']} failed: {finished.get('error')}")
        from pydantic import TypeAdapter

        from strata.labels import AnyPrediction

        parse = TypeAdapter(AnyPrediction)
        found = {c: parse.validate_python(v) for c, v in finished["result"]["predictions"].items()}
        missing = [c for c in checksums if c not in found]
        if missing:
            raise StageError(f"The host's catalog does not know {len(missing)} of the samples.")
        return found, "remote"

    cache = PredictionCache.beside(context.store)
    found = cache.get(request.run_id, checksums, digests)
    todo = [s for s in samples if s.checksum not in found]
    if todo:
        scored = handlers.predict(
            PredictRequest(
                run_id=request.run_id,
                paths=[Path(request.dataset_dir) / s.path for s in todo],
                features=[s.features for s in todo],
            ),
            context.store,
        )
        made = {s.checksum: out.value for s, out in zip(todo, scored, strict=True)}
        cache.put(request.run_id, made, digests)
        found.update(made)
    return found, "local"


def _manifest(directory: Path) -> Manifest:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise StageError(f"{directory} is not a materialised dataset: no {MANIFEST_NAME}.")
    return Manifest.model_validate_json(path.read_text())


STAGES = (
    Stage("train", "1", (DATASET_DIR,), RUN, train),
    # 2: exact_match rather than accuracy, and per-class scores. A record
    # written by 1 has the old shape, so it is not handed back as this one's.
    Stage("evaluate", "2", (DATASET_DIR, RUN), METRICS, evaluate),
)

__all__ = [
    "DATASET_DIR",
    "METRICS",
    "RUN",
    "STAGES",
    "ClassScore",
    "Context",
    "DatasetIdentity",
    "EvaluateRecord",
    "EvaluateRequest",
    "Host",
    "StageError",
    "TrainRecord",
    "TrainStageRequest",
    "evaluate",
    "train",
]
