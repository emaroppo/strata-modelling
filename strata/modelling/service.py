"""Training as a service: the same handler, reached over a wire.

The one place in this package allowed to import ``catalog``, and the reason
the training core is not. The core takes a directory and a manifest and
nothing else, which is what makes a dataset version a portable artifact and
lets the core be tested against a fixture directory with no database in
sight. Materialising is the shell's job.

What crosses the wire is a dataset id, not a directory. The host has the
index and the bucket; shipping it a gigabyte of images it could fetch
itself would be paying the network to avoid using the network.

Two things live here rather than with the caller, because they need the run
store and the run store lives beside the checkpoints:

**Choosing a parent.** Warm-starting means loading a checkpoint, so only
the machine holding checkpoints can decide which one, or verify the choice.

**Recording the run.** A run names a checkpoint by path. Recorded anywhere
else it would name a file that machine does not have.
"""

import os
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from .handlers import train as run_train
from .registry import available
from .requests import Run, TrainRequest
from .runs import RunStore


class ServiceError(Exception):
    """A request this host cannot honour."""


class RoundRequest(BaseModel):
    """One training round, as asked for from somewhere else."""

    #: A dataset in the catalog this host is configured against. Its version
    #: is already frozen — the caller made it — so two hosts materialising
    #: the same id get the same bytes.
    dataset_id: int
    #: A registered short name. File references are refused: see
    #: :func:`check_servable`.
    model: str
    params: dict = Field(default_factory=dict)
    #: Cold start, ignoring whatever this host last trained on this dataset.
    fresh: bool = False


class RoundResponse(BaseModel):
    run: Run
    metrics: dict[str, float] = Field(default_factory=dict)
    #: How the dataset was obtained, so a caller can tell a cache hit from a
    #: transfer without reading the host's logs.
    materialised: int = 0


def check_servable(model: str) -> None:
    """Refuse a model this host cannot honestly resolve.

    A file reference names a path on the caller's machine. Importing it here
    would either fail confusingly or, worse, find a different file with the
    same name — and a service that imports whatever path a caller names is a
    different thing from a service that serves what it has.
    """
    if ":" not in model:
        return
    raise ServiceError(
        f"{model!r} is a direct reference to code on the caller's machine, and "
        f"this host cannot resolve it. Either register the model as a "
        f"'strata.models' entry point on this host and ask for it by name "
        f"(available here: {', '.join(sorted(available())) or 'none'}), or run "
        f"the round locally, where file references still work."
    )


def run_round(
    request: RoundRequest,
    catalog,
    store: RunStore,
    datasets: Path,
    cache: Path | None = None,
    on_progress=None,
) -> RoundResponse:
    """Materialise a dataset version and train from it.

    ``datasets`` is where versions are written. Named by dataset and
    version, so two rounds over the same version share one directory rather
    than each fetching a copy.
    """
    from strata.catalog import MANIFEST_NAME, Manifest

    check_servable(request.model)

    name, version = catalog.dataset_named(request.dataset_id)
    target = Path(datasets) / name / f"v{version:03d}"

    materialised = 0
    if not (target / MANIFEST_NAME).exists():
        staging = Path(datasets) / name / "pending"
        if staging.exists():
            shutil.rmtree(staging)
        counted = {"n": 0}

        def tick(done: int, total: int) -> None:
            counted["n"] = done
            if on_progress is not None:
                on_progress(done, total)

        catalog.materialise(request.dataset_id, staging, on_progress=tick, cache=cache)
        staging.rename(target)
        materialised = counted["n"]

    manifest = Manifest.model_validate_json((target / MANIFEST_NAME).read_text())
    previous = None if request.fresh else store.latest(manifest.dataset)

    run = run_train(
        TrainRequest(
            dataset_dir=target,
            model=request.model,
            params=request.params,
            parent_run_id=previous.id if previous else None,
        ),
        store,
    )
    return RoundResponse(
        run=run, metrics=_metrics_of(store, run.id), materialised=materialised
    )


def _metrics_of(store: RunStore, run_id: int) -> dict[str, float]:
    from sqlalchemy import select

    from . import tables as t

    with store.engine.connect() as conn:
        return {
            row.name: row.value
            for row in conn.execute(
                select(t.metric.c.name, t.metric.c.value).where(t.metric.c.run_id == run_id)
            )
        }


def _required(name: str, why: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ServiceError(f"${name} is not set. {why}")
    return value


def build():
    """The app, from the environment. Raises if anything essential is absent."""
    from fastapi import Depends, FastAPI, Header, HTTPException

    from strata.catalog import Catalog

    token = _required(
        "STRATA_MODELLING_TOKEN",
        "This host trains what it is asked to; an unauthenticated one trains "
        "what anyone asks it to.",
    )
    catalog_url = _required(
        "STRATA_CATALOG_URL",
        "The host materialises the dataset it is asked to train on.",
    )
    root = Path(os.environ.get("STRATA_MODELLING_ROOT", "modelling"))
    datasets = root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    store = RunStore.local(root / "runs")
    cache = Path(os.environ["STRATA_BLOBS_CACHE"]) if os.environ.get("STRATA_BLOBS_CACHE") else None

    def catalog_for() -> Catalog:
        # Per request rather than once: a long-lived connection to a database
        # on another machine outlives its usefulness, and training rounds are
        # far enough apart that reconnecting costs nothing.
        return Catalog.connect(catalog_url, _blobs())

    def _blobs():
        endpoint = os.environ.get("STRATA_S3_ENDPOINT")
        if not endpoint:
            from strata.catalog import LocalBackend

            return LocalBackend(Path(_required(
                "STRATA_BLOBS_ROOT", "With no S3 endpoint the host reads files."
            )))
        import boto3
        from botocore.config import Config

        from strata.catalog.s3 import S3Backend

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("STRATA_S3_ACCESS_KEY") or None,
            aws_secret_access_key=os.environ.get("STRATA_S3_SECRET_KEY") or None,
            region_name=os.environ.get("STRATA_S3_REGION", "garage"),
            config=Config(s3={"addressing_style": "path"}),
        )
        return S3Backend(client, bucket=_required(
            "STRATA_S3_BUCKET", "An endpoint without a bucket names nothing."
        ))

    def authorise(authorization: str = Header(default="")) -> None:
        import hmac

        offered = authorization.removeprefix("Bearer ").strip()
        # Constant time: a comparison that returns early leaks the token a
        # character at a time to anyone able to measure it
        if not hmac.compare_digest(offered, token):
            raise HTTPException(status_code=403, detail="Bad or missing token.")

    app = FastAPI(title="strata modelling", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/models", dependencies=[Depends(authorise)])
    def models() -> dict:
        """What this host can serve, read from what is installed."""
        return {"models": available()}

    @app.post("/round", dependencies=[Depends(authorise)])
    def round_(request: RoundRequest) -> RoundResponse:
        try:
            return run_round(request, catalog_for(), store, datasets, cache)
        except ServiceError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    @app.get("/runs/{run_id}", dependencies=[Depends(authorise)])
    def get_run(run_id: int) -> dict:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"No run {run_id}")
        return {"run": run.model_dump(mode="json"), "metrics": _metrics_of(store, run_id)}

    return app


def main() -> None:
    """Entry point. Serves until stopped."""
    import sys

    import uvicorn

    try:
        app = build()
    except ServiceError as e:
        print(f"strata-modelling: {e}", file=sys.stderr)
        raise SystemExit(2) from None

    uvicorn.run(
        app,
        host=os.environ.get("STRATA_SERVE_HOST", "0.0.0.0"),  # noqa: S104
        port=int(os.environ.get("STRATA_SERVE_PORT", "8082")),
    )
