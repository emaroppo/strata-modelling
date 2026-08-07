"""Training and prediction: the handler both transports call.

The in-process path and the HTTP adapter call *these functions*, rather than
HTTP wrapping a second implementation. So validation happens once, an error
reads the same locally and remotely, and the failure mode where something
works on a laptop and 400s against the GPU host cannot arise.
"""

import inspect
import json
from pathlib import Path

from sqlalchemy import update

from strata.labels import Choices, ClassificationSchema

from . import tables as t
from .model import Example, Model
from .registry import ModelError, absolute, resolve
from .requests import Prediction, PredictRequest, Run, TrainRequest
from .runs import RunStore

MANIFEST_NAME = "manifest.json"


class TrainingError(Exception):
    """A training job that cannot be run as requested."""


def train(request: TrainRequest, store: RunStore) -> Run:
    """Train from a materialised dataset and record the run."""
    manifest = _read_manifest(request.dataset_dir)
    schema = ClassificationSchema.model_validate(manifest["label_schema"])
    model_cls = resolve(request.model, root=request.dataset_dir)

    if model_cls.task != schema.task:
        raise TrainingError(
            f"Model {request.model!r} handles '{model_cls.task}' but this label "
            f"set is '{schema.task}'"
        )

    model: Model = _construct(model_cls, request.model, request.params)
    classes = list(schema.classes)
    parent = _warm_start(model, request, store, classes)

    train_examples, val_examples = _examples(request.dataset_dir, manifest)
    if not train_examples:
        raise TrainingError(f"{request.dataset_dir} has no training samples")

    metrics = model.finetune(train_examples, classes, val_examples or None)

    run = store.record(
        Run(
            id=0,
            parent_run_id=parent.id if parent else None,
            dataset=manifest["dataset"],
            dataset_version=manifest["version"],
            label_set=manifest["label_set"],
            # Anchored, so predicting or warm-starting from this run later
            # does not depend on the dataset directory still being there
            model=absolute(request.model, request.dataset_dir),
            model_version=model_cls.version,
            params=request.params,
            classes=classes,
        ),
        metrics,
    )
    checkpoint = store.checkpoint_path(run.id)
    model.save(checkpoint)
    return _attach_checkpoint(store, run, checkpoint)


def predict(request: PredictRequest, store: RunStore) -> list[Prediction]:
    """Run a recorded checkpoint over some paths.

    Predictions are returned rather than written. Persisting them is the
    caller's business, and doing it here would put a catalog dependency back
    into the training core.
    """
    run = store.get(request.run_id)
    if run is None:
        raise TrainingError(f"No run with id {request.run_id}")
    if run.checkpoint is None or not Path(run.checkpoint).exists():
        raise TrainingError(f"Run {run.id} has no checkpoint on disk")

    model: Model = _construct(resolve(run.model), run.model, run.params)
    model.load(Path(run.checkpoint))
    outputs = model.predict(list(request.paths))
    return [
        Prediction(path=path, value=value)
        for path, value in zip(request.paths, outputs, strict=True)
    ]


# ----------------------------------------------------------------------


def _construct(model_cls: type[Model], name: str, params: dict) -> Model:
    """Build the model, turning a bad parameter into something readable.

    In process a wrong keyword is a TypeError from deep inside the
    constructor. Over a wire it would be a 500 with a traceback, so it is
    named here instead — one error, the same either side.
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


def _read_manifest(directory: Path) -> dict:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise TrainingError(f"No {MANIFEST_NAME} in {directory}")
    return json.loads(path.read_text())


def _examples(directory: Path, manifest: dict) -> tuple[list[Example], list[Example]]:
    train_examples, val_examples = [], []
    for sample in manifest["samples"]:
        if sample.get("value") is None:
            # Skipped: reviewed, nothing applicable, not training data
            continue
        example = Example(
            path=Path(directory) / sample["path"],
            target=Choices.model_validate(sample["value"]),
        )
        (val_examples if sample.get("val") else train_examples).append(example)
    return train_examples, val_examples


def _warm_start(
    model: Model, request: TrainRequest, store: RunStore, classes: list[str]
) -> Run | None:
    if request.parent_run_id is None:
        return None
    parent = store.get(request.parent_run_id)
    if parent is None:
        raise TrainingError(f"No run with id {request.parent_run_id} to continue from")
    if parent.model_version != type(model).version:
        # Output neurons map to the class list by position, so a checkpoint
        # from a different version of the model is not merely stale — loading
        # it corrupts silently instead of failing.
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


def _attach_checkpoint(store: RunStore, run: Run, checkpoint: Path) -> Run:
    with store.engine.begin() as conn:
        conn.execute(
            update(t.run).where(t.run.c.id == run.id).values(checkpoint=str(checkpoint))
        )
    return run.model_copy(update={"checkpoint": checkpoint})


__all__ = ["ModelError", "TrainingError", "predict", "train"]
