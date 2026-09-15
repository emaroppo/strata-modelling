"""Running a round or a scoring pass on this host: materialise, train, record; fetch, predict."""

from pathlib import Path

from strata.labels import AnyPrediction, feature_digest

from ..handlers import train as run_train
from ..requests import PredictRequest, TrainRequest
from ..store.runs import RunStore
from .checks import check_catalog, check_dataset, check_features, check_servable, check_split
from .wire import (
    PredictionRequest,
    PredictionResponse,
    RoundRequest,
    RoundResponse,
)


def run_round(
    request: RoundRequest,
    catalog,
    store: RunStore,
    datasets: Path,
    cache: Path | None = None,
    report=None,
) -> RoundResponse:
    """Materialise a dataset version and train from it.

    ``datasets`` is where versions are written, named by dataset and
    version. See ``docs/adr/0007``.
    """
    from strata.catalog import ensure_materialised

    check_servable(request.model)

    check_catalog(request.catalog_id, catalog.id)
    check_dataset(request, catalog.datasets.named(request.dataset_id))
    specs = check_features(request)

    def tick(done: int, total: int) -> None:
        if report is not None:
            report("materialising", done, total)

    # The same reuse rule the laptop uses, because it is the same function,
    # and with the project's features. docs/adr/0026
    result = ensure_materialised(
        catalog, request.dataset_id, datasets, features=specs, on_progress=tick, cache=cache
    )
    manifest, target, materialised = result.manifest, result.directory, result.fetched

    # The caller's sides, when they are not the version's own, applied to a
    # copy beside it by the same code the local split stage uses. docs/adr/0025
    if request.split is not None:
        sides = check_split(request.split, manifest)
        if sides is not None:
            from strata.catalog.stages import apply_sides
            from strata.common.canonical import short_hash

            asked = request.split
            tag = short_hash(
                {"sides": asked.sides, "val": asked.val_ratio, "holdout": asked.holdout_ratio},
                length=12,
            )
            target, manifest = apply_sides(
                target,
                manifest,
                sides,
                val_ratio=asked.val_ratio,
                holdout_ratio=asked.holdout_ratio,
                tag=tag,
            )

    previous = (
        None
        if request.fresh
        else store.latest(
            manifest.dataset, manifest.catalog_id, since_version=manifest.sides_from_version
        )
    )
    params = {**request.params, **request.fresh_params} if previous is None else request.params

    # Said before training rather than after. docs/adr/0031
    if report is not None:
        report("training")

    def epoch(done: int, total: int, metrics: dict) -> None:
        if report is None:
            return
        note = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        report(f"training ({note})" if note else "training", done, total)

    run = run_train(
        TrainRequest(
            dataset_dir=target,
            model=request.model,
            params=params,
            parent_run_id=previous.id if previous else None,
            experiment_id=request.experiment_id,
        ),
        store,
        on_epoch=epoch,
    )
    return RoundResponse(run=run, metrics=metrics_of(store, run.id), materialised=materialised)


def run_prediction(
    request: PredictionRequest,
    catalog,
    store: RunStore,
    cache: Path,
    report=None,
) -> PredictionResponse:
    """Score samples by content, fetching whatever bytes are missing.

    See ``docs/adr/0006``.
    """
    from ..handlers import predict as run_predict
    from ..store.predictions import PredictionCache

    # Kept beside the runs. docs/adr/0006
    known = PredictionCache.beside(store)
    # Keyed on the same three inputs the caller's cache is; the digest is
    # computed here from what arrived rather than sent. docs/adr/0006
    digests = {c: feature_digest(request.features.get(c, {})) for c in request.checksums}
    already = known.get(request.run_id, request.checksums, digests)
    wanted = [c for c in request.checksums if c not in already]

    def fetching(done: int, total: int) -> None:
        if report is not None:
            report("fetching", done, total)

    paths = catalog.ensure_cached(wanted, cache, on_progress=fetching)
    unknown = [c for c in wanted if c not in paths]

    made: dict[str, AnyPrediction] = {}
    if paths:
        if report is not None:
            report("predicting", 0, len(paths))

        def scoring(done: int, total: int) -> None:
            if report is not None:
                report("predicting", done, total)

        ordered = list(paths)
        outputs = run_predict(
            PredictRequest(
                run_id=request.run_id,
                paths=[paths[c] for c in ordered],
                features=[request.features.get(c, {}) for c in ordered],
            ),
            store,
            on_batch=scoring,
        )
        made = {c: o.value for c, o in zip(ordered, outputs, strict=True)}
        known.put(request.run_id, made, digests)

    return PredictionResponse(predictions={**already, **made}, unknown=unknown)


def metrics_of(store: RunStore, run_id: str) -> dict[str, float]:
    from sqlalchemy import select

    from ..store import tables as t

    with store.engine.connect() as conn:
        return {
            row.name: row.value
            for row in conn.execute(
                select(t.metric.c.name, t.metric.c.value).where(t.metric.c.run_id == run_id)
            )
        }
