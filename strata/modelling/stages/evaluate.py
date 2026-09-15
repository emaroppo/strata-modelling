"""The evaluate stage: one side of a directory scored with a recorded run, by one implementation."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from strata.labels import Choices, feature_digest

from .. import handlers
from ..remote.client import RemoteError
from ..remote.wire import PredictionRequest
from ..requests import PredictRequest
from ..store.predictions import PredictionCache
from ._context import Context, StageError, Strict, Where, _manifest

# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------


class EvaluateRequest(Strict):
    """Score one side of a directory with a run, by one implementation."""

    run_id: str
    dataset_dir: Path
    #: ``holdout`` is what a study reports on; ``val`` is what it selects on
    #: (``docs/adr/0003``).
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
    where: Where
    #: Samples with an answer on that side; the denominator.
    samples: int
    #: ``exact_match`` is the share of samples whose asserted set of classes
    #: is exactly the model's; ``precision``, ``recall`` and ``f1`` are micro,
    #: over every class assertion. See ``docs/adr/0035``.
    metrics: dict[str, float]
    per_class: dict[str, ClassScore] = Field(default_factory=dict)


def evaluate(request: EvaluateRequest, context: Context) -> EvaluateRecord:
    manifest = _manifest(request.dataset_dir)
    task = manifest.label_schema.task
    if task != "classification":
        # Refused rather than approximated. docs/adr/0035
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
        truth = set(_choices(sample.value).values)
        guessed = set(_choices(predictions[sample.checksum]).values)
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


def _choices(value: object) -> Choices:
    """A classification answer, which is what this stage scores; refused otherwise."""
    if not isinstance(value, Choices):
        raise StageError(f"evaluate scores classification only; got {type(value).__name__}")
    return value


def _class_score(tally: list[int]) -> ClassScore:
    precision, recall, f1 = _prf(*tally)
    return ClassScore(precision=precision, recall=recall, f1=f1, support=tally[0] + tally[2])


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


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

        from strata.labels import AnyPrediction

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
