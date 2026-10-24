"""Training, prediction and generation: the handler both transports call.

The in-process path and the HTTP adapter call *these functions*, so
validation happens once. See ``docs/adr/0007``.
"""

import hashlib
import inspect
import math
import shutil
from pathlib import Path

from strata.common.canonical import content_hash
from strata.contracts import MANIFEST_NAME, Manifest, ManifestFormatError

from .model import Example, LabelModel, MediaModel, Model
from .plugins.registry import ModelError, absolute, resolve
from .requests import (
    Checkpoint,
    GeneratedOutput,
    GenerateRequest,
    PredictRequest,
    Run,
    ScoredPath,
    TrainRequest,
)
from .store.runs import CHECKPOINT_NAME, FINAL, RunStore, Seen, checkpoint_digest, now


class TrainingError(Exception):
    """A training job that cannot be run as requested."""


def train(request: TrainRequest, store: RunStore, on_epoch=None) -> Run:
    """Train from a materialised dataset and record the run."""
    manifest = _read_manifest(request.dataset_dir)
    schema = manifest.label_schema
    model_cls = resolve(request.model, root=request.dataset_dir)
    # What is checked before the round follows what the model makes.
    # docs/adr/0054
    media = issubclass(model_cls, MediaModel)

    if issubclass(model_cls, MediaModel):
        _check_sample_types(model_cls, request.model, manifest)
    elif issubclass(model_cls, LabelModel) and model_cls.label_type != schema.label_type:
        raise TrainingError(
            f"Model {request.model!r} handles '{model_cls.label_type}' but this label "
            f"set is '{schema.label_type}'"
        )

    # Built before its requirements are read. docs/adr/0014
    model = _construct(model_cls, request.model, request.params)

    undeclared = (
        [c for c in model.requires_classes if c not in schema.classes]
        if isinstance(model, LabelModel)
        else []
    )
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

    if isinstance(model, LabelModel):
        try:
            model.requires_schema(schema)
        except ValueError as e:
            # Refused before the round. docs/adr/0014
            raise TrainingError(
                f"Model {request.model!r} cannot be trained on this label set: {e}"
            ) from None
    # A media model has no classes: what it learns is the samples. docs/adr/0054
    classes = [] if media else list(schema.classes)
    parent = _warm_start(model, request, store, classes)

    train_examples, val_examples = examples(request.dataset_dir, manifest, targets=not media)
    if not train_examples:
        raise TrainingError(f"{request.dataset_dir} has no training samples")
    train_examples = nested_subset(train_examples, request.train_fraction, request.subset_seed)

    # Kept as well as forwarded; a model that never calls back records no
    # curve. docs/adr/0005
    curve: list[tuple[int, dict[str, float]]] = []

    def collect(done: int, total: int, reported: dict[str, float]) -> bool | None:
        curve.append((done, dict(reported)))
        # The caller's answer goes back to the model: a request to stop early
        # is the caller's to make and the model's to honour, or not
        return on_epoch(done, total, reported) if on_epoch is not None else None

    saving = _Saving(model, store)
    try:
        if isinstance(model, MediaModel):
            metrics = model.finetune(
                train_examples, val_examples or None, collect, **_offered(model.finetune, saving)
            )
        else:
            # Nothing else resolves. docs/adr/0054
            assert isinstance(model, LabelModel)
            metrics = model.finetune(
                train_examples,
                classes,
                val_examples or None,
                collect,
                **_offered(model.finetune, saving),
            )
        # How much was trained on, on the run itself: a learning curve's x axis
        metrics["train_samples"] = float(len(train_examples))
        # What it actually ran with, defaults included, where it says.
        # docs/adr/0054
        resolved = model.resolved_params()
        params = request.params if resolved is None else resolved
        if not saving.saved:
            # A model that never asked is kept once, at the end, as it always
            # was. docs/adr/0054
            saving(FINAL)

        return store.record(
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
                params=params,
                classes=classes,
            ),
            metrics,
            curve,
            # What this run saw: the side of every sample it trained from. docs/adr/0005
            saw=[Seen(s.checksum, s.split, s.batch, s.reviewed) for s in manifest.samples],
            # Moved under the run once it has an id. docs/adr/0054
            checkpoints=saving.saved,
        )
    finally:
        shutil.rmtree(saving.directory, ignore_errors=True)


def predict(request: PredictRequest, store: RunStore, on_batch=None) -> list[ScoredPath]:
    """Run a recorded checkpoint over some paths.

    Predictions are returned rather than written. See ``docs/adr/0006``.
    """
    run, checkpoint = _recorded(store, request.run_id, request.checkpoint)
    model_cls = resolve(run.model)
    if not issubclass(model_cls, LabelModel):
        raise TrainingError(
            f"Run {run.id} was trained by {run.model!r}, a media model: it generates "
            f"rather than predicts. Ask it to generate."
        )

    model = _construct(model_cls, run.model, run.params)
    model.load(_on_disk(run, checkpoint))
    outputs = model.predict(list(request.paths), on_batch, features=request.features or None)
    return [
        ScoredPath(path=path, value=value)
        for path, value in zip(request.paths, outputs, strict=True)
    ]


def generate(request: GenerateRequest, store: RunStore, on_batch=None) -> list[GeneratedOutput]:
    """Generate from a media model's recorded checkpoint, one output per input.

    Outputs are returned rather than written: the caller knows which
    samples it passed, and commits them as a generate run. Each says the
    settings digest and seed that made it, positionally. See
    ``docs/adr/0053`` and ``docs/adr/0054``.
    """
    run, checkpoint = _recorded(store, request.run_id, request.checkpoint)
    model_cls = resolve(run.model)
    if not issubclass(model_cls, MediaModel):
        raise TrainingError(
            f"Run {run.id} was trained by {run.model!r}, a label model: it predicts "
            f"rather than generates."
        )

    model = _construct(model_cls, run.model, run.params)
    model.load(_on_disk(run, checkpoint))
    made = model.generate(
        list(request.inputs), list(request.settings), list(request.seeds), on_batch
    )
    if len(made) != len(request.inputs):
        raise TrainingError(
            f"{run.model!r} made {len(made)} output(s) for {len(request.inputs)} input(s); "
            f"generate returns one per input, in order."
        )
    return [
        GeneratedOutput(
            data=out.data,
            ended=out.ended,
            # Over the canonical form, so equal settings agree wherever
            # they were written. docs/adr/0017
            settings_digest=content_hash(settings),
            seed=seed,
            checkpoint=checkpoint.name,
            checkpoint_digest=checkpoint.digest,
        )
        for out, settings, seed in zip(made, request.settings, request.seeds, strict=True)
    ]


# ----------------------------------------------------------------------


def _construct[M: Model](model_cls: type[M], name: str, params: dict) -> M:
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


def nested_subset(train: list[Example], fraction: float, seed: int) -> list[Example]:
    """The first ``fraction`` of ``train`` in an order fixed by ``seed``.

    Ranked by a hash of the seed and each file's name, never by position, so
    the subsets for one seed are nested — 25% inside 50% inside 100% — and
    the same whatever order the manifest lists them in. At least one sample.
    """
    if fraction >= 1.0:
        return train
    ranked = sorted(
        train, key=lambda e: hashlib.sha256(f"{seed}:{Path(e.path).name}".encode()).hexdigest()
    )
    return ranked[: max(1, math.ceil(fraction * len(train)))]


def examples(
    directory: Path, manifest: Manifest, targets: bool = True
) -> tuple[list[Example], list[Example]]:
    """What a model is handed from a materialised dataset: training, then validation.

    Read through the manifest's own definition. Public so every label type
    is tested through it. See ``docs/adr/0004``.

    Without ``targets``, for a media model, an example carries none: the
    file is what it learns to make, and its features the input. See
    ``docs/adr/0054``.
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
            target=sample.value if targets else None,
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
    checkpoint = parent.checkpoint_named(request.parent_checkpoint)
    if request.parent_checkpoint is not None:
        # Named, it has to be there: a warm start from the wrong weights is
        # worse than none. docs/adr/0054
        if checkpoint is None:
            raise TrainingError(_no_such_checkpoint(parent, request.parent_checkpoint))
        model.load(_on_disk(parent, checkpoint))
    elif checkpoint is not None and checkpoint.path is not None and checkpoint.path.exists():
        model.load(checkpoint.path)
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


class _Saving:
    """The ``save_checkpoint`` a model is handed, and what it saved.

    The run's id is minted only once training is done, so each checkpoint
    is saved aside and moved under the run as it is recorded. See
    ``docs/adr/0054``.
    """

    def __init__(self, model: Model, store: RunStore):
        self.model = model
        self.directory = store.staging()
        self.saved: list[Checkpoint] = []

    def __call__(self, name: str) -> None:
        if not CHECKPOINT_NAME.fullmatch(name):
            raise TrainingError(
                f"A checkpoint name becomes a file name, so it is letters, digits, "
                f"'.', '_' and '-', starting with a letter or digit, at most 64; "
                f"{name!r} is not."
            )
        if any(c.name == name for c in self.saved):
            raise TrainingError(
                f"A checkpoint named {name!r} was already saved in this run. Name "
                f"each one apart, as epoch-2 is from epoch-3."
            )
        path = self.directory / name
        self.model.save(path)
        self.saved.append(
            Checkpoint(name=name, path=path, digest=checkpoint_digest(path), saved_at=now())
        )


def _offered(finetune, saving: _Saving) -> dict[str, _Saving]:
    """``save_checkpoint``, for a ``finetune`` that takes it, and nothing for one that does not.

    So no model has to change for it. See ``docs/adr/0054``.
    """
    takes = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        or (p.name == "save_checkpoint" and p.kind is not inspect.Parameter.POSITIONAL_ONLY)
        for p in inspect.signature(finetune).parameters.values()
    )
    return {"save_checkpoint": saving} if takes else {}


def _recorded(store: RunStore, run_id: str, name: str | None) -> tuple[Run, Checkpoint]:
    """A run, and the checkpoint a request names or its last."""
    run = store.get(run_id)
    if run is None:
        raise TrainingError(f"No run with id {run_id}")
    checkpoint = run.checkpoint_named(name)
    if checkpoint is None:
        if name is not None:
            raise TrainingError(_no_such_checkpoint(run, name))
        raise TrainingError(f"Run {run.id} has no checkpoint on disk")
    return run, checkpoint


def _on_disk(run: Run, checkpoint: Checkpoint) -> Path:
    if checkpoint.path is None or not checkpoint.path.exists():
        raise TrainingError(f"Run {run.id} has no checkpoint {checkpoint.name!r} on disk")
    return checkpoint.path


def _no_such_checkpoint(run: Run, name: str) -> str:
    kept = ", ".join(c.name for c in run.checkpoints) or "none"
    return f"Run {run.id} has no checkpoint named {name!r}. It kept: {kept}."


def _check_sample_types(model_cls: type[MediaModel], name: str, manifest: Manifest) -> None:
    """Refuse samples that are not of the type a media model trains on.

    Every sample of the manifest, held out or not. A type's subtypes are
    accepted, and the plain case of a media accepts the whole media, by
    the rule a catalog's selection uses. See ``docs/adr/0052`` and
    ``docs/adr/0054``.
    """
    wanted = model_cls.trains_on
    if wanted is None:
        raise TrainingError(
            f"Model {name!r} is a media model and names no sample type in `trains_on`. "
            f"Name the type it learns from, such as SFTTextExample."
        )
    from strata.contracts.sample_types import PLAIN

    media, subtype = wanted.media, wanted.subtype()
    want = f"{media}/{subtype}"

    def holds(stored: str) -> bool:
        their_media, _, their_subtype = stored.partition("/")
        if their_media != media:
            return False
        return (
            subtype == PLAIN or their_subtype == subtype or their_subtype.startswith(subtype + "/")
        )

    untyped = sum(1 for s in manifest.samples if s.type is None)
    if untyped:
        raise TrainingError(
            f"Model {name!r} trains on {want} samples, and {untyped} sample(s) in this "
            f"manifest do not say their type: it was written before manifests did. "
            f"Materialise the dataset again."
        )
    other = sorted({s.type for s in manifest.samples if s.type is not None and not holds(s.type)})
    if other:
        raise TrainingError(
            f"Model {name!r} trains on {want} samples, and this dataset holds "
            f"{', '.join(other)}. Freeze a version of {want} samples, or train a "
            f"model that learns from these."
        )


__all__ = ["ModelError", "TrainingError", "generate", "predict", "train"]
