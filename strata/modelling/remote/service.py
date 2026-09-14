"""The HTTP app: routes over the jobs, the checks and the run store, built from the environment."""

import os
from pathlib import Path

from ..plugins.registry import available
from ..store.runs import RunStore
from .checks import check_catalog, check_dataset, check_features
from .jobs import BusyError, Job, Jobs
from .rounds import metrics_of, run_prediction, run_round
from .wire import (
    PROTOCOL,
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
    from fastapi import Depends, FastAPI, Header, HTTPException

    from strata.catalog import Catalog, CatalogError

    token = _required(
        "STRATA_MODELLING_TOKEN",
        "This host trains what it is asked to; an unauthenticated one trains "
        "what anyone asks it to.",
    )
    from strata.catalog.config import CatalogConfigError, host_catalog, open_catalog

    # The same config.toml format the CLI reads, from the file
    # $STRATA_CONFIG names — on this host, the CLI's own. Its default is the
    # catalog this host trains from, so switching is changing that default
    # and restarting.
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
        # Per request rather than once: a long-lived connection to a database
        # on another machine outlives its usefulness, and training rounds are
        # far enough apart that reconnecting costs nothing. Opened by the
        # same code the CLI uses, so the two cannot read a catalog differently.
        return open_catalog(catalog_config)

    def authorise(authorization: str = Header(default="")) -> None:
        import hmac

        offered = authorization.removeprefix("Bearer ").strip()
        # Constant time: a comparison that returns early leaks the token a
        # character at a time to anyone able to measure it
        if not hmac.compare_digest(offered, token):
            raise HTTPException(status_code=403, detail="Bad or missing token.")

    def spoken(x_strata_protocol: str = Header(default="")) -> None:
        # 426: the request is well formed, and needs a matching release to
        # be understood. Checked on every call rather than once, because a
        # laptop can be upgraded, or downgraded, between two of them.
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
        # Memoised: a catalog's identity does not change under a running
        # host, and the alternative is a connection per submitted round
        # spent asking a question with one answer.
        if "id" not in served:
            served["id"] = catalog_for().id
        return str(served["id"])

    def run_one(request: RoundRequest, report) -> RoundResponse:
        return run_round(request, catalog_for(), store, datasets, cache, report=report)

    jobs = Jobs(run_one)
    app = FastAPI(title="strata modelling", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz() -> dict:
        # Unauthenticated, so a laptop can check it speaks this host's
        # protocol, and is on its catalog, before sending anything at all
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
        try:
            # Before accepting, not inside the job: a caller that gets a 202
            # for a round which cannot run learns nothing until it polls
            check_catalog(request.catalog_id, served_catalog_id())
            try:
                found = catalog_for().datasets.named(request.dataset_id)
            except CatalogError as e:
                raise ServiceError(str(e)) from None
            check_dataset(request, found)
            check_features(request)
            return jobs.submit(request)
        except BusyError as e:
            # 409 rather than 400: the request is fine, the host is not free
            raise HTTPException(status_code=409, detail=str(e)) from None
        except ServiceError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    @app.post("/predict", dependencies=[Depends(authorise), Depends(spoken)], status_code=202)
    def predict_(request: PredictionRequest) -> Job:
        """Accept a scoring job. Same queue as training: both need the GPU."""
        if cache is None:
            raise HTTPException(
                status_code=400,
                detail="No $STRATA_BLOBS_CACHE, so this host has nowhere to put "
                "the samples it would have to fetch to score them.",
            )

        cache_dir = cache

        def run_it(req, report):
            return run_prediction(req, catalog_for(), store, cache_dir, report)

        try:
            return jobs.submit(request, runner=run_it)
        except BusyError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        except ServiceError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    @app.get("/jobs/{job_id}", dependencies=[Depends(authorise), Depends(spoken)])
    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            # A restart loses running jobs; a completed round is in the run
            # store, which is the half worth recovering
            raise HTTPException(
                status_code=404,
                detail=f"No job {job_id}. Jobs do not survive a restart of this host; "
                f"a round that finished is in the run store.",
            )
        return job

    @app.get("/runs/latest", dependencies=[Depends(authorise), Depends(spoken)])
    def latest_run(dataset: str, catalog: str | None = None) -> dict:
        """The newest run over a dataset, in this host's numbering.

        A caller cannot work this out for itself: a run is minted where it
        happens, and the caller's own store holds the runs made there, not
        the ones made here.
        """
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
    from strata.common.service import serve

    serve(build, prog="strata-modelling", port=8082, error=ServiceError)
