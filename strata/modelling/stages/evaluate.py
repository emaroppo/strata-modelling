"""The evaluate stage: one side of a directory scored with a recorded run, by the tasks asked.

A task is strata-evaluation's: found by name or by a project's own file, it
reads one label type and asks one question of it, and what it records
carries the identity of the code that produced it. Predictions come from the
cache or the modelling host; the scoring always runs here. See
``docs/adr/0035`` and ``docs/adr/0042``.

Every checkpoint the run kept is scored, each on its own, and the record
says which checkpoint each score belongs to. A media model's run is scored
by generating: one output from each sample's input, which a generation task
reads beside the side's targets and the training side's. The outputs stay
in the evaluation's result, never the catalog. See ``docs/adr/0054``.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, TypeAdapter

from strata.common.canonical import content_hash
from strata.contracts import AnyPrediction, feature_digest
from strata.evaluation.identity import Identity
from strata.evaluation.registry import PluginError, resolve
from strata.evaluation.tasks import (
    BaseTask,
    Document,
    Generated,
    Generation,
    GenerationTask,
    Task,
    TaskError,
    TaskScore,
)
from strata.evaluation.text import UnreadableDocument, read_document

from .. import handlers
from ..model import GenerationInput, MediaModel
from ..plugins.registry import ModelError
from ..plugins.registry import resolve as resolve_model
from ..remote.client import RemoteError
from ..remote.wire import GenerationInputRef, GenerationRequest, PredictionRequest
from ..requests import GeneratedOutput, GenerateRequest, PredictRequest, Run
from ..store.predictions import PredictionCache
from ._context import Context, Keyed, StageError, Strict, Where, _manifest

# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------

#: What a label set is scored by when a request names no task.
DEFAULT_TASKS: dict[str, list[str]] = {
    "classification": ["classify"],
    "span": ["entities"],
}

#: What a media model's outputs are scored by when a request names no task,
#: by the media it makes. See ``docs/adr/0054``.
DEFAULT_GENERATION_TASKS: dict[str, list[str]] = {
    "text": ["text-generation"],
}


class TaskRef(Strict):
    """A task to run: its name or ``file.py:Class``, its parameters, and what it was checked as.

    ``identities`` is the task's own identity followed by those of the plugins
    its parameters name, as resolved when the request was made. Given, it is
    held to: code that changed between then and now is refused rather than
    scored under the old identity. See ``docs/adr/0042``.
    """

    ref: str
    params: dict[str, Any] = Field(default_factory=dict)
    identities: list[Identity] | None = None


class EvaluateRequest(Keyed):
    """Score one side of a directory with a run, by the tasks named."""

    later = frozenset({"checkpoints", "settings", "seed"})

    run_id: str
    dataset_dir: Path
    #: ``holdout`` is what a study reports on; ``val`` is what it selects on
    #: (``docs/adr/0003``).
    side: Literal["val", "holdout"] = "holdout"
    #: None is the label type's default: classify, or entities. For a media
    #: model, its media's: text-generation.
    tasks: list[TaskRef] | None = None
    #: Which of the run's checkpoints to score, by name. None is every one
    #: it kept. See ``docs/adr/0054``.
    checkpoints: list[str] | None = None
    #: A media model's generation settings, over its own defaults. Empty is
    #: its defaults. Every output is made with the same.
    settings: dict = Field(default_factory=dict)
    #: What each output's seed is drawn from, with its sample's checksum.
    #: See :func:`sample_seed`.
    seed: int = 0


class ClassScore(Strict):
    precision: float
    recall: float
    f1: float
    #: How many held-out samples assert the class.
    support: int


class GeneratedText(Strict):
    """One output a media model made to be scored, and what made it besides the checkpoint."""

    #: The sample whose input it was made from, and whose file is its target.
    checksum: str
    seed: int
    #: Whether it stopped on its own rather than at the token limit; None
    #: where the model did not say.
    ended: bool | None = None
    output: str


class CheckpointScores(Strict):
    """One checkpoint's scores, and what a media model generated for them."""

    #: None where a modelling host older than checkpoints answered from the
    #: run's last without naming it.
    name: str | None
    digest: str | None = None
    #: Each task's score, by the task's name, with what produced it.
    scores: dict[str, TaskScore] = Field(default_factory=dict)
    #: A media model's outputs, in sample order, positional with any
    #: per-output detail in ``scores``. Empty when kept in a file instead.
    outputs: list[GeneratedText] = Field(default_factory=list)
    #: The file they were kept in, by name, under the context's ``outputs``:
    #: one JSON line per output. Named by its content, so a record that is
    #: found again names the same file.
    outputs_file: str | None = None


class EvaluateRecord(Strict):
    run_id: str
    side: str
    where: Where
    #: Samples with an answer on that side; the denominator.
    samples: int
    #: Each task's score, by the task's name, with what produced it: the
    #: last checkpoint scored, which is the run's last unless the request
    #: names others. Every checkpoint's is in ``checkpoints``.
    scores: dict[str, TaskScore] = Field(default_factory=dict)
    #: The classify task's numbers, where it ran, as records before tasks
    #: had them: ``exact_match`` and micro ``precision``, ``recall``, ``f1``.
    #: Kept so a reader of older records reads these the same way.
    metrics: dict[str, float] = Field(default_factory=dict)
    per_class: dict[str, ClassScore] = Field(default_factory=dict)
    #: The checkpoint ``scores`` belong to. None on a record from before
    #: checkpoints, and where a host did not name it. See ``docs/adr/0054``.
    checkpoint: str | None = None
    #: Every checkpoint scored, oldest first. Empty on a record from before
    #: checkpoints, which scored the run's last.
    checkpoints: list[CheckpointScores] = Field(default_factory=list)


def makes(model: str) -> str | None:
    """What a model makes: the media of the type a media model trains on, or None for labels.

    Read from the class, so the run's model has to resolve here, wherever
    it was trained. See ``docs/adr/0054``.
    """
    try:
        model_cls = resolve_model(model)
    except ModelError as e:
        raise StageError(
            f"Cannot tell whether {model!r} predicts labels or generates media, since it "
            f"does not resolve here: {e}"
        ) from None
    if not issubclass(model_cls, MediaModel):
        return None
    if model_cls.trains_on is None:
        raise StageError(f"Media model {model!r} names no sample type in `trains_on`.")
    return model_cls.trains_on.media


def resolved(
    ref: TaskRef, schema, media: str | None = None
) -> tuple[BaseTask, Any, list[Identity]]:
    """The task a ref names, its parameters, and the identities it resolves to now.

    ``media`` is what the run's model generates, or None for a label model.
    A label task is checked against the label set and a generation task
    against the media, and either one is refused for the other kind of
    model. Refused, before anything is scored, when the task does not
    exist, will not take its parameters, cannot read this label set or
    media, or names a plugin that does not resolve.
    """
    try:
        cls, identity = resolve("task", ref.ref)
        task = cls()
        params = task.params(ref.params)
        if isinstance(task, GenerationTask):
            if media is None:
                raise TaskError(
                    f"Task {task.name!r} scores what a media model generates, and this run's "
                    f"model predicts labels. Name a task that reads {schema.label_type} "
                    f"label sets, or leave tasks out for its default."
                )
            task.check(params, media)
        elif isinstance(task, Task):
            if media is not None:
                raise TaskError(
                    f"Task {task.name!r} scores a label model's predictions, and this run's "
                    f"model generates {media}. Name a task that reads generated {media}, "
                    f"or leave tasks out for its default."
                )
            task.check(params, schema)
        identities = [identity] + [resolve(kind, r)[1] for kind, r in task.requires(params)]
    except (PluginError, TaskError) as e:
        raise StageError(str(e)) from None
    return task, params, identities


def sample_seed(seed: int, checksum: str) -> int:
    """The seed one sample's output is generated with, under a stage's ``seed``.

    Drawn from the sample's checksum rather than its position, so a sample
    keeps its seed whatever else is on the side and whatever order the
    manifest lists it in, and every checkpoint is asked with the same one:
    two checkpoints' outputs then differ by their weights, not their luck.
    Below 2**31, which every framework's seeding takes.
    """
    digest = hashlib.sha256(f"{seed}:{checksum}".encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


def evaluate(request: EvaluateRequest, context: Context) -> EvaluateRecord:
    manifest = _manifest(request.dataset_dir)
    label_type = manifest.label_schema.label_type
    run = _run(request.run_id, context)
    # The kind of model decides what the tasks read. docs/adr/0054
    media = makes(run.model)
    refs = request.tasks
    if refs is None:
        if media is not None:
            if media not in DEFAULT_GENERATION_TASKS:
                raise StageError(
                    f"No task scores generated {media} by default yet; name one that does."
                )
            refs = [TaskRef(ref=name) for name in DEFAULT_GENERATION_TASKS[media]]
        elif label_type not in DEFAULT_TASKS:
            # Refused rather than approximated. docs/adr/0035
            raise StageError(
                f"No task scores a {label_type} label set by default yet; name one "
                f"that does, or leave it unscored. Box-level scoring is still to come."
            )
        else:
            refs = [TaskRef(ref=name) for name in DEFAULT_TASKS[label_type]]

    tasks = []
    for ref in refs:
        task, params, identities = resolved(ref, manifest.label_schema, media)
        if ref.identities is not None and ref.identities != identities:
            raise StageError(
                f"Task {ref.ref!r} was checked as {_sources(ref.identities)} and resolves "
                f"now as {_sources(identities)}: the code changed in between. Check the "
                f"experiment again."
            )
        tasks.append((task, params, identities[0]))
    names = [task.name for task, _, _ in tasks]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise StageError(f"Task(s) named more than once: {', '.join(twice)}.")

    samples = [s for s in manifest.samples if s.split == request.side and s.value is not None]
    if not samples:
        raise StageError(
            f"Nothing to score: {request.dataset_dir} has no answered sample on the "
            f"{request.side!r} side."
        )
    checkpoints = _chosen(run, request.checkpoints)

    where: Where = "remote" if context.host is not None else "local"
    if media is None:
        scored = _scored_labels(request, context, tasks, samples, checkpoints)
    else:
        training = [s for s in manifest.samples if s.split == "train" and s.value is not None]
        scored = _scored_generations(request, context, tasks, samples, training, checkpoints)

    last = scored[-1]
    classified = last.scores.get("classify")
    return EvaluateRecord(
        run_id=request.run_id,
        side=request.side,
        where=where,
        samples=len(samples),
        scores=last.scores,
        metrics=dict(classified.metrics) if classified else {},
        per_class=(
            {
                name: ClassScore(
                    precision=c["precision"],
                    recall=c["recall"],
                    f1=c["f1"],
                    support=int(c["support"]),
                )
                for name, c in classified.per_class.items()
            }
            if classified
            else {}
        ),
        checkpoint=last.name,
        checkpoints=scored,
    )


def _sources(identities: list[Identity]) -> str:
    return ", ".join(f"{i.name} {i.version} ({i.source})" for i in identities)


def _run(run_id: str, context: Context) -> Run:
    """The run as whoever holds it records it: this store, or the modelling host."""
    if context.host is not None:
        client = context.client(context.host.url, context.host.token)
        found = client.run(run_id)
        if found is None:
            raise StageError(f"The modelling host holds no run {run_id!r}.")
        return Run.model_validate(found["run"])
    if context.store is None:
        raise StageError("Scoring here needs a run store; none was given.")
    run = context.store.get(run_id)
    if run is None:
        raise StageError(f"No run {run_id!r} in the run store.")
    return run


#: A checkpoint to score: its name and digest, or (None, None) for the run's
#: last where its holder does not name checkpoints.
Chosen = tuple[str | None, str | None]


def _chosen(run: Run, named: list[str] | None) -> list[Chosen]:
    """The checkpoints to score, oldest first: every one the run kept, or those named.

    A run from a host older than checkpoints lists none, and is scored once,
    from whatever that host calls its last. See ``docs/adr/0054``.
    """
    if not run.checkpoints:
        if named:
            raise StageError(
                f"Run {run.id} lists no checkpoints, so none can be named: its holder "
                f"is older than checkpoints. Leave checkpoints out to score its last."
            )
        return [(None, None)]
    kept = [c.name for c in run.checkpoints]
    if named is not None:
        unknown = [n for n in named if n not in kept]
        if unknown:
            raise StageError(
                f"Run {run.id} kept no checkpoint named {', '.join(map(repr, unknown))}. "
                f"It kept: {', '.join(kept)}."
            )
        if not named:
            raise StageError("checkpoints names none; leave it out to score every one.")
    return [(c.name, c.digest) for c in run.checkpoints if named is None or c.name in named]


# ----------------------------------------------------------------------
# a label model: predicted, then scored against the answers
# ----------------------------------------------------------------------


def _scored_labels(request: EvaluateRequest, context: Context, tasks, samples, checkpoints):
    texts: dict[str, str] = {}
    if any(isinstance(task, Task) and task.needs_text for task, _, _ in tasks):
        try:
            texts = {s.checksum: read_document(Path(request.dataset_dir) / s.path) for s in samples}
        except UnreadableDocument as e:
            raise StageError(str(e)) from None

    scored = []
    for name, digest in checkpoints:
        predictions = _predictions(request, context, samples, name)
        # Every sample here has an answer; the filter says so to the type checker
        documents = [
            Document(s.value, predictions[s.checksum], texts.get(s.checksum))
            for s in samples
            if s.value is not None
        ]
        scores: dict[str, TaskScore] = {}
        for task, params, identity in tasks:
            try:
                scores[task.name] = task.recorded(documents, params, identity)
            except TaskError as e:
                raise StageError(str(e)) from None
        scored.append(CheckpointScores(name=name, digest=digest, scores=scores))
    return scored


def _predictions(request: EvaluateRequest, context: Context, samples, checkpoint: str | None):
    """What a checkpoint says about each sample, from the cache where it can be.

    The cache is keyed on checkpoint, bytes and features. See ``docs/adr/0006``.
    """
    digests = {s.checksum: feature_digest(s.features) for s in samples}
    checksums = [s.checksum for s in samples]

    if context.host is not None:
        client = context.client(context.host.url, context.host.token)
        job = client.predict(
            PredictionRequest(
                run_id=request.run_id,
                checkpoint=checkpoint,
                checksums=checksums,
                features={s.checksum: s.features for s in samples if s.features},
            )
        )
        finished = client.follow(job["id"], on_state=context.on_state)
        if finished.get("state") != "done":
            raise RemoteError(f"The host reports job {job['id']} failed: {finished.get('error')}")
        parse = TypeAdapter(AnyPrediction)
        found = {c: parse.validate_python(v) for c, v in finished["result"]["predictions"].items()}
        missing = [c for c in checksums if c not in found]
        if missing:
            raise StageError(f"The host's catalog does not know {len(missing)} of the samples.")
        return found

    store = context.store
    if store is None:
        raise StageError("Scoring here needs a run store; none was given.")
    cache = PredictionCache.beside(store)
    # Named once, so the lookup and the pass agree. docs/adr/0054
    checkpoint = checkpoint or cache.last_checkpoint(request.run_id)
    found = cache.get(request.run_id, checksums, digests, checkpoint=checkpoint)
    todo = [s for s in samples if s.checksum not in found]
    if todo:
        scored = handlers.predict(
            PredictRequest(
                run_id=request.run_id,
                checkpoint=checkpoint or None,
                paths=[Path(request.dataset_dir) / s.path for s in todo],
                features=[s.features for s in todo],
            ),
            store,
        )
        made = {s.checksum: out.value for s, out in zip(todo, scored, strict=True)}
        cache.put(request.run_id, made, digests, checkpoint=checkpoint)
        found.update(made)
    return found


# ----------------------------------------------------------------------
# a media model: generated from each input, then scored as outputs
# ----------------------------------------------------------------------


def _scored_generations(
    request: EvaluateRequest, context: Context, tasks, samples, training, checkpoints
):
    """Each checkpoint's outputs, one per sample's input, scored beside the targets.

    The input is the sample's features, as it was in training; samples by
    role are not given yet. Every output is made with the request's
    settings, and each with its sample's seed. The targets are the side's
    files and ``training`` the training side's, read as documents are. See
    ``docs/adr/0054``.
    """
    try:
        targets = [read_document(Path(request.dataset_dir) / s.path) for s in samples]
        trained = [read_document(Path(request.dataset_dir) / s.path) for s in training]
    except UnreadableDocument as e:
        raise StageError(str(e)) from None
    inputs = [GenerationInput(values=dict(s.features)) for s in samples]
    seeds = [sample_seed(request.seed, s.checksum) for s in samples]

    scored = []
    for name, digest in checkpoints:
        made = _generated(request, context, name, inputs, seeds)
        if len(made) != len(samples):
            raise StageError(
                f"{len(made)} output(s) came back for {len(samples)} input(s); there is one "
                f"per input, in order."
            )
        outputs = [
            GeneratedText(
                checksum=s.checksum,
                seed=out.seed,
                ended=out.ended,
                # Read as a document is, so a stray byte is marked rather than
                # refused. strata.evaluation.text
                output=out.data.decode("utf-8", errors="replace"),
            )
            for s, out in zip(samples, made, strict=True)
        ]
        generation = Generation(
            outputs=[Generated(output=o.output, ended=o.ended) for o in outputs],
            targets=targets,
            training=trained,
        )
        scores: dict[str, TaskScore] = {}
        for task, params, identity in tasks:
            try:
                scores[task.name] = task.recorded(generation, params, identity)
            except TaskError as e:
                raise StageError(str(e)) from None
        kept = _kept(outputs, context.outputs)
        scored.append(
            CheckpointScores(
                name=name or (made[0].checkpoint if made else None),
                digest=digest or (made[0].checkpoint_digest if made else None),
                scores=scores,
                outputs=[] if kept else outputs,
                outputs_file=kept,
            )
        )
    return scored


def _generated(
    request: EvaluateRequest,
    context: Context,
    checkpoint: str | None,
    inputs: list[GenerationInput],
    seeds: list[int],
) -> list[GeneratedOutput]:
    settings = [dict(request.settings) for _ in inputs]
    if context.host is not None:
        client = context.client(context.host.url, context.host.token)
        job = client.generate(
            GenerationRequest(
                run_id=request.run_id,
                checkpoint=checkpoint,
                inputs=[GenerationInputRef(values=i.values) for i in inputs],
                settings=settings,
                seeds=seeds,
            )
        )
        finished = client.follow(job["id"], on_state=context.on_state)
        if finished.get("state") != "done":
            raise RemoteError(f"The host reports job {job['id']} failed: {finished.get('error')}")
        # As JSON, where an output's bytes are base64
        parse = TypeAdapter(list[GeneratedOutput])
        return parse.validate_json(json.dumps(finished["result"]["outputs"]))

    store = context.store
    if store is None:
        raise StageError("Scoring here needs a run store; none was given.")
    try:
        return handlers.generate(
            GenerateRequest(
                run_id=request.run_id,
                checkpoint=checkpoint,
                inputs=inputs,
                settings=settings,
                seeds=seeds,
            ),
            store,
        )
    except handlers.TrainingError as e:
        raise StageError(str(e)) from None


def _kept(outputs: list[GeneratedText], directory: Path | None) -> str | None:
    """Write ``outputs`` under ``directory``, one JSON line each, and return the file's name.

    Named by the content, so writing the same outputs again is the same
    file, and a file is whole or absent (``docs/adr/0026``). None, writing
    nothing, without a directory.
    """
    if directory is None:
        return None
    lines = [o.model_dump(mode="json") for o in outputs]
    name = f"{content_hash(lines)}.jsonl"
    path = Path(directory) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{name}.{os.getpid()}")
        partial.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines))
        partial.replace(path)
    return name
