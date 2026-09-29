"""The evaluate stage: one side of a directory scored with a recorded run, by one implementation."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from strata.contracts import Choices, feature_digest
from strata.evaluation import Tally
from strata.evaluation.tasks import classify

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

    scores = classify.score(
        [_choices(s.value) for s in samples],
        [_choices(predictions[s.checksum]) for s in samples],
    )
    return EvaluateRecord(
        run_id=request.run_id,
        side=request.side,
        where=where,
        samples=scores.samples,
        metrics={
            "exact_match": scores.exact_match,
            "precision": scores.micro.precision,
            "recall": scores.micro.recall,
            "f1": scores.micro.f1,
        },
        per_class={name: _class_score(tally) for name, tally in scores.per_class.items()},
    )


def _choices(value: object) -> Choices:
    """A classification answer, which is what this stage scores; refused otherwise."""
    if not isinstance(value, Choices):
        raise StageError(f"evaluate scores classification only; got {type(value).__name__}")
    return value


def _class_score(tally: Tally) -> ClassScore:
    return ClassScore(
        precision=tally.precision, recall=tally.recall, f1=tally.f1, support=tally.support
    )


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
