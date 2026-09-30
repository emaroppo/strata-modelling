"""The evaluate stage: one side of a directory scored with a recorded run, by the tasks asked.

A task is strata-evaluation's: found by name or by a project's own file, it
reads one label type and asks one question of it, and what it records
carries the identity of the code that produced it. Predictions come from the
cache or the modelling host; the scoring always runs here. See
``docs/adr/0035`` and ``docs/adr/0042``.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from strata.contracts import feature_digest
from strata.evaluation.identity import Identity
from strata.evaluation.registry import PluginError, resolve
from strata.evaluation.tasks import Document, Task, TaskError, TaskScore
from strata.evaluation.text import UnreadableDocument, read_document

from .. import handlers
from ..remote.client import RemoteError
from ..remote.wire import PredictionRequest
from ..requests import PredictRequest
from ..store.predictions import PredictionCache
from ._context import Context, StageError, Strict, Where, _manifest

# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------

#: What a label set is scored by when a request names no task.
DEFAULT_TASKS: dict[str, list[str]] = {
    "classification": ["classify"],
    "span": ["entities"],
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


class EvaluateRequest(Strict):
    """Score one side of a directory with a run, by the tasks named."""

    run_id: str
    dataset_dir: Path
    #: ``holdout`` is what a study reports on; ``val`` is what it selects on
    #: (``docs/adr/0003``).
    side: Literal["val", "holdout"] = "holdout"
    #: None is the label type's default: classify, or entities.
    tasks: list[TaskRef] | None = None


class ClassScore(Strict):
    precision: float
    recall: float
    f1: float
    #: How many held-out samples assert the class.
    support: int


class EvaluateRecord(Strict):
    run_id: str
    side: str
    where: Where
    #: Samples with an answer on that side; the denominator.
    samples: int
    #: Each task's score, by the task's name, with what produced it.
    scores: dict[str, TaskScore] = Field(default_factory=dict)
    #: The classify task's numbers, where it ran, as records before tasks
    #: had them: ``exact_match`` and micro ``precision``, ``recall``, ``f1``.
    #: Kept so a reader of older records reads these the same way.
    metrics: dict[str, float] = Field(default_factory=dict)
    per_class: dict[str, ClassScore] = Field(default_factory=dict)


def resolved(ref: TaskRef, schema) -> tuple[Task, Any, list[Identity]]:
    """The task a ref names, its parameters, and the identities it resolves to now.

    Refused, before anything is scored, when the task does not exist, will
    not take its parameters, cannot read this label set, or names a plugin
    that does not resolve.
    """
    try:
        cls, identity = resolve("task", ref.ref)
        task = cls()
        params = task.params(ref.params)
        task.check(params, schema)
        identities = [identity] + [resolve(kind, r)[1] for kind, r in task.requires(params)]
    except (PluginError, TaskError) as e:
        raise StageError(str(e)) from None
    return task, params, identities


def evaluate(request: EvaluateRequest, context: Context) -> EvaluateRecord:
    manifest = _manifest(request.dataset_dir)
    label_type = manifest.label_schema.label_type
    refs = request.tasks
    if refs is None:
        if label_type not in DEFAULT_TASKS:
            # Refused rather than approximated. docs/adr/0035
            raise StageError(
                f"No task scores a {label_type} label set by default yet; name one "
                f"that does, or leave it unscored. Box-level scoring is still to come."
            )
        refs = [TaskRef(ref=name) for name in DEFAULT_TASKS[label_type]]

    tasks = []
    for ref in refs:
        task, params, identities = resolved(ref, manifest.label_schema)
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
    predictions, where = _predictions(request, context, samples)

    texts: dict[str, str] = {}
    if any(task.needs_text for task, _, _ in tasks):
        try:
            texts = {s.checksum: read_document(Path(request.dataset_dir) / s.path) for s in samples}
        except UnreadableDocument as e:
            raise StageError(str(e)) from None
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

    classified = scores.get("classify")
    return EvaluateRecord(
        run_id=request.run_id,
        side=request.side,
        where=where,
        samples=len(samples),
        scores=scores,
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
    )


def _sources(identities: list[Identity]) -> str:
    return ", ".join(f"{i.name} {i.version} ({i.source})" for i in identities)


def _predictions(request: EvaluateRequest, context: Context, samples) -> tuple[dict, Where]:
    """What the run says about each sample, from the cache where it can be.

    The cache is keyed on checkpoint, bytes and features. See ``docs/adr/0006``.
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

        from strata.contracts import AnyPrediction

        parse = TypeAdapter(AnyPrediction)
        found = {c: parse.validate_python(v) for c, v in finished["result"]["predictions"].items()}
        missing = [c for c in checksums if c not in found]
        if missing:
            raise StageError(f"The host's catalog does not know {len(missing)} of the samples.")
        return found, "remote"

    store = context.store
    if store is None:
        raise StageError("Scoring here needs a run store; none was given.")
    cache = PredictionCache.beside(store)
    found = cache.get(request.run_id, checksums, digests)
    todo = [s for s in samples if s.checksum not in found]
    if todo:
        scored = handlers.predict(
            PredictRequest(
                run_id=request.run_id,
                paths=[Path(request.dataset_dir) / s.path for s in todo],
                features=[s.features for s in todo],
            ),
            store,
        )
        made = {s.checksum: out.value for s, out in zip(todo, scored, strict=True)}
        cache.put(request.run_id, made, digests)
        found.update(made)
    return found, "local"
