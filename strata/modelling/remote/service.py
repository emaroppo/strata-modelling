"""The HTTP app: routes over the jobs, the checks and the run store, built from the environment."""

import hmac
import os
from pathlib import Path

from strata.common.service import serve

from ..plugins.registry import available
from ..store.runs import RunStore
from .checks import check_catalog, check_dataset, check_features
from .jobs import BusyError, Job, Jobs
from .rounds import metrics_of, run_generation, run_prediction, run_round
from .wire import (
    PROTOCOL,
    GenerationRequest,
    PredictionRequest,
    RoundRequest,
    RoundResponse,
    ServiceError,
)


def _required(name: str, why: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ServiceError(f"${name} is not set. {why}")
    return value


def build():
    """The app, from the environment. Raises if anything essential is absent."""
    # The service extra: fastapi, and the catalog this layer alone may know (docs/adr/0007).
    from fastapi import Depends, FastAPI, Header, HTTPException

    from strata.catalog import Catalog, CatalogError

    token = _required(
        "STRATA_MODELLING_TOKEN",
        "This host trains what it is asked to; an unauthenticated one trains "
        "what anyone asks it to.",
    )
    from strata.catalog.config import CatalogConfigError, host_catalog, open_catalog

    # The same config file the CLI reads; its default catalog is the one served. docs/adr/0019
    try:
        catalog_name, catalog_config = host_catalog()
    except CatalogConfigError as e:
        raise ServiceError(str(e)) from None
    root = Path(os.environ.get("STRATA_MODELLING_ROOT", "modelling"))
    datasets = root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    store = RunStore.local(root / "runs")
    cache = Path(os.environ["STRATA_BLOBS_CACHE"]) if os.environ.get("STRATA_BLOBS_CACHE") else None

    def catalog_for() -> Catalog:
        # Per request rather than once. docs/adr/0019
        return open_catalog(catalog_config)

    def authorise(authorization: str = Header(default="")) -> None:
        offered = authorization.removeprefix("Bearer ").strip()
        # Constant time, so the comparison leaks nothing about the token.
        if not hmac.compare_digest(offered, token):
            raise HTTPException(status_code=403, detail="Bad or missing token.")

    def spoken(x_strata_protocol: str = Header(default="")) -> None:
        # 426: well formed, and needs a matching release to be understood. docs/adr/0007
        if x_strata_protocol != str(PROTOCOL):
            asked = f"is protocol {x_strata_protocol}" if x_strata_protocol else "names none"
            raise HTTPException(
                status_code=426,
                detail=f"This host speaks protocol {PROTOCOL}, and the request {asked}. "
                f"The laptop and this host must run matching releases: upgrade "
                f"whichever is older.",
            )

    served: dict[str, str | None] = {}

    def served_catalog_id() -> str:
        # Memoised: a catalog's identity does not change under a running host.
        if "id" not in served:
            served["id"] = catalog_for().id
        return str(served["id"])

    def run_one(request: RoundRequest, report) -> RoundResponse:
        return run_round(request, catalog_for(), store, datasets, cache, report=report)

    jobs = Jobs(run_one)

    def _submit(request, runner=None, check=lambda: None) -> Job:
        """``check``, then the job queued; a busy host is 409, a refusal 400 (``docs/adr/0007``)."""
        try:
            check()
            return jobs.submit(request, runner=runner)
        except BusyError as e:
            # 409: the request is fine, the host is not free.
            raise HTTPException(status_code=409, detail=str(e)) from None
        except ServiceError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    app = FastAPI(title="strata modelling", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz() -> dict:
        # Unauthenticated, so a caller can check before sending anything. docs/adr/0007
        served = {"name": catalog_name, "id": None}
        try:
            served["id"] = served_catalog_id()
        except Exception as e:  # an unreachable index is worth reporting, not a 500
            served["error"] = f"{type(e).__name__}: {e}"
        return {"ok": True, "protocol": PROTOCOL, "catalog": served}

    @app.get("/models", dependencies=[Depends(authorise), Depends(spoken)])
    def models() -> dict:
        """What this host can serve, read from what is installed."""
        return {"models": available()}

    @app.post("/round", dependencies=[Depends(authorise), Depends(spoken)], status_code=202)
    def round_(request: RoundRequest) -> Job:
        """Accept a round and return the job. It runs after this responds."""

        def checked() -> None:
            # Checked before accepting, not inside the job. docs/adr/0007
            check_catalog(request.catalog_id, served_catalog_id())
            try:
                found = catalog_for().datasets.named(request.dataset_id)
            except CatalogError as e:
                raise ServiceError(str(e)) from None
            check_dataset(request, found)
            check_features(request)

        return _submit(request, check=checked)

    @app.post("/predict", dependencies=[Depends(authorise), Depends(spoken)], status_code=202)
    def predict_(request: PredictionRequest) -> Job:
        """Accept a scoring job. Same queue as training (``docs/adr/0007``)."""
        if cache is None:
            raise HTTPException(
                status_code=400,
                detail="No $STRATA_BLOBS_CACHE, so this host has nowhere to put "
                "the samples it would have to fetch to score them.",
            )

        cache_dir = cache

        def run_it(req, report):
            return run_prediction(req, catalog_for(), store, cache_dir, report)

        return _submit(request, run_it)

    @app.post("/generate", dependencies=[Depends(authorise), Depends(spoken)], status_code=202)
    def generate_(request: GenerationRequest) -> Job:
        """Accept a generation job. Same queue as training (``docs/adr/0054``)."""
        if cache is None and any(i.samples for i in request.inputs):
            raise HTTPException(
                status_code=400,
                detail="No $STRATA_BLOBS_CACHE, so this host has nowhere to put "
                "the samples the inputs name.",
            )
        cache_dir = cache or root / "blobs"

        def run_it(req, report):
            return run_generation(req, catalog_for(), store, cache_dir, report)

        return _submit(request, run_it)

    @app.get("/jobs/{job_id}", dependencies=[Depends(authorise), Depends(spoken)])
    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            # Jobs are in memory; a finished round is in the run store. docs/adr/0007
            raise HTTPException(
                status_code=404,
                detail=f"No job {job_id}. Jobs do not survive a restart of this host; "
                f"a round that finished is in the run store.",
            )
        return job

    @app.get("/runs/latest", dependencies=[Depends(authorise), Depends(spoken)])
    def latest_run(dataset: str, catalog: str | None = None) -> dict:
        """The newest run over a dataset, in this host's numbering. See ``docs/adr/0005``."""
        run = store.latest(dataset, catalog)
        if run is None:
            raise HTTPException(status_code=404, detail=f"No runs over {dataset!r} here")
        return {"run": run.model_dump(mode="json"), "metrics": metrics_of(store, run.id)}

    @app.get("/runs/{run_id}", dependencies=[Depends(authorise), Depends(spoken)])
    def get_run(run_id: str) -> dict:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"No run {run_id}")
        return {"run": run.model_dump(mode="json"), "metrics": metrics_of(store, run_id)}

    return app


def main() -> None:
    """Entry point. Serves until stopped."""
    serve(build, prog="strata-modelling", port=8082, error=ServiceError)
