"""Training and prediction: the handler both transports call.

The in-process path and the HTTP adapter call *these functions*, so
validation happens once. See ``docs/adr/0007``.
"""

import inspect
from pathlib import Path

from sqlalchemy import update

from strata.labels import MANIFEST_NAME, Manifest, ManifestFormatError

from .model import Example, Model
from .plugins.registry import ModelError, absolute, resolve
from .requests import PredictRequest, Run, ScoredPath, TrainRequest
from .store import tables as t
from .store.runs import RunStore, Seen


class TrainingError(Exception):
    """A training job that cannot be run as requested."""


def train(request: TrainRequest, store: RunStore, on_epoch=None) -> Run:
    """Train from a materialised dataset and record the run."""
    manifest = _read_manifest(request.dataset_dir)
    schema = manifest.label_schema
    model_cls = resolve(request.model, root=request.dataset_dir)

    if model_cls.task != schema.task:
        raise TrainingError(
            f"Model {request.model!r} handles '{model_cls.task}' but this label "
            f"set is '{schema.task}'"
        )

    # Built before its requirements are read. docs/adr/0014
    model: Model = _construct(model_cls, request.model, request.params)

    undeclared = [c for c in model.requires_classes if c not in schema.classes]
    if undeclared:
        # An implicit class must be declared. docs/adr/0014
        raise TrainingError(
            f"Model {request.model!r} predicts {', '.join(undeclared)}, which this "
            f"label set does not declare (it has: {', '.join(schema.classes)}). "
            f"Either declare it, or use a model without an implicit negative class."
        )

    declared: set[str] = {str(f["name"]) for f in manifest.features if f.get("name")}
    unmet = [f for f in model.requires_features if f not in declared]
    if unmet:
        # Refused before the round. docs/adr/0014
        raise TrainingError(
            f"Model {request.model!r} needs feature(s) {', '.join(unmet)}, which "
            f"this dataset does not carry (it has: {', '.join(sorted(declared)) or 'none'}). "
            f"Declare them under [[data.features]] and materialise again."
        )

    try:
        model.requires_schema(schema)
    except ValueError as e:
        # Refused before the round. docs/adr/0014
        raise TrainingError(
            f"Model {request.model!r} cannot be trained on this label set: {e}"
        ) from None
    classes = list(schema.classes)
    parent = _warm_start(model, request, store, classes)

    train_examples, val_examples = examples(request.dataset_dir, manifest)
    if not train_examples:
        raise TrainingError(f"{request.dataset_dir} has no training samples")

    # Kept as well as forwarded; a model that never calls back records no
    # curve. docs/adr/0005
    curve: list[tuple[int, dict[str, float]]] = []

    def collect(done: int, total: int, reported: dict[str, float]) -> bool | None:
        curve.append((done, dict(reported)))
        # The caller's answer goes back to the model: a request to stop early
        # is the caller's to make and the model's to honour, or not
        return on_epoch(done, total, reported) if on_epoch is not None else None

    metrics = model.finetune(train_examples, classes, val_examples or None, collect)

    run = store.record(
        Run(
            # Minted by the store, where the run happened
            id="",
            parent_run_id=parent.id if parent else None,
            dataset=manifest.dataset,
            dataset_version=manifest.version,
            # From the manifest rather than the request. docs/adr/0005
            catalog_id=manifest.catalog_id,
            experiment_id=request.experiment_id,
            label_set=manifest.label_set,
            # Anchored. docs/adr/0005
            model=absolute(request.model, request.dataset_dir),
            model_version=model_cls.version,
            params=request.params,
            classes=classes,
        ),
        metrics,
        curve,
        # What this run saw: the side of every sample it trained from. docs/adr/0005
        saw=[Seen(s.checksum, s.split, s.batch, s.reviewed) for s in manifest.samples],
    )
    checkpoint = store.checkpoint_path(run.id)
    model.save(checkpoint)
    return _attach_checkpoint(store, run, checkpoint)


def predict(request: PredictRequest, store: RunStore, on_batch=None) -> list[ScoredPath]:
    """Run a recorded checkpoint over some paths.

    Predictions are returned rather than written. See ``docs/adr/0006``.
    """
    run = store.get(request.run_id)
    if run is None:
        raise TrainingError(f"No run with id {request.run_id}")
    if run.checkpoint is None or not Path(run.checkpoint).exists():
        raise TrainingError(f"Run {run.id} has no checkpoint on disk")

    model: Model = _construct(resolve(run.model), run.model, run.params)
    model.load(Path(run.checkpoint))
    outputs = model.predict(list(request.paths), on_batch, features=request.features or None)
    return [
        ScoredPath(path=path, value=value)
        for path, value in zip(request.paths, outputs, strict=True)
    ]


# ----------------------------------------------------------------------


def _construct(model_cls: type[Model], name: str, params: dict) -> Model:
    """Build the model, turning a bad parameter into a ``TrainingError``.

    See ``docs/adr/0007``.
    """
    try:
        return model_cls(**params)
    except TypeError as exc:
        accepted = [
            p
            for p in inspect.signature(model_cls).parameters
            if p not in ("self", "args", "kwargs")
        ]
        raise TrainingError(
            f"{name!r} will not accept these parameters: {exc}. "
            f"It takes: {', '.join(accepted) or 'none'}."
        ) from exc


def _read_manifest(directory: Path) -> Manifest:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise TrainingError(f"No {MANIFEST_NAME} in {directory}")
    try:
        return Manifest.model_validate_json(path.read_text())
    except ManifestFormatError as e:
        # Refused rather than rebuilt. docs/adr/0004
        raise TrainingError(f"{directory}: {e}") from None


def examples(directory: Path, manifest: Manifest) -> tuple[list[Example], list[Example]]:
    """What a model is handed from a materialised dataset: training, then validation.

    Read through the manifest's own definition. Public so every label type
    is tested through it. See ``docs/adr/0004``.
    """
    train_examples, val_examples = [], []
    for sample in manifest.samples:
        if sample.value is None:
            # Skipped: reviewed, nothing applicable, not training data
            continue
        if sample.split == "holdout":
            # Never handed to a model, not even as validation. docs/adr/0003
            continue
        example = Example(
            path=Path(directory) / sample.path,
            target=sample.value,
            features=sample.features,
        )
        (val_examples if sample.split == "val" else train_examples).append(example)
    return train_examples, val_examples


def _warm_start(
    model: Model, request: TrainRequest, store: RunStore, classes: list[str]
) -> Run | None:
    if request.parent_run_id is None:
        return None
    parent = store.get(request.parent_run_id)
    if parent is None:
        raise TrainingError(f"No run with id {request.parent_run_id} to continue from")
    _refuse_a_different_model(parent, type(model))
    if parent.model_version != type(model).version:
        # A warm start across a model version is refused. docs/adr/0005
        raise TrainingError(
            f"Run {parent.id} was trained by {parent.model!r} version "
            f"{parent.model_version}, and the installed one is version "
            f"{type(model).version}. Train a fresh run rather than continuing."
        )
    if classes[: len(parent.classes)] != parent.classes:
        raise TrainingError(
            f"Run {parent.id} trained on {parent.classes}, and the label set is "
            f"now {classes}. Classes are append-only because a checkpoint maps "
            f"output neurons to them by position; reordering or removing one "
            f"invalidates it."
        )
    if parent.checkpoint and Path(parent.checkpoint).exists():
        model.load(Path(parent.checkpoint))
    return parent


def _refuse_a_different_model(parent: Run, model_cls: type[Model]) -> None:
    """Refuse to continue from a checkpoint another model wrote.

    Compared by class name; a parent reference that no longer resolves is
    left alone. See ``docs/adr/0025``.
    """
    try:
        was = resolve(parent.model)
    except ModelError:
        return
    if was.__name__ == model_cls.__name__:
        return
    raise TrainingError(
        f"Run {parent.id} was trained by {was.__name__} and this is "
        f"{model_cls.__name__}. A checkpoint is one model's weights in that "
        f"model's layout; loading it into another is not a warm start. Train "
        f"a fresh run instead."
    )


def _attach_checkpoint(store: RunStore, run: Run, checkpoint: Path) -> Run:
    with store.engine.begin() as conn:
        conn.execute(update(t.run).where(t.run.c.id == run.id).values(checkpoint=str(checkpoint)))
    return run.model_copy(update={"checkpoint": checkpoint})


__all__ = ["ModelError", "TrainingError", "predict", "train"]
