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

**A round is submitted, not awaited.** Training is minutes, and the caller
is a laptop that closes. Holding a connection open for the whole run makes
the round only as reliable as the network and the lid, so the request
returns a job and the caller polls. Losing the connection then costs
nothing: the work carries on here, and the job is still there to ask about.

One round at a time, refused rather than queued. A second training job on
one GPU does not run slower, it runs out of memory and takes the first one
with it.

**Both sides speak one protocol.** The laptop and this host are separate
releases once the packages are, and a field one side added and the other
ignores fails silently — which is how remote rounds would have dropped
features. So ``/healthz`` states :data:`PROTOCOL`, the laptop checks it
before sending anything, and every other request names it or is refused.
The number goes up only when an older side would misread a newer one; a
field added with a default does not need it.
"""

import os
import threading
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from strata.labels import AnyPrediction, Prediction, feature_digest

from .handlers import train as run_train
from .registry import available
from .requests import PredictRequest, Run, TrainRequest
from .runs import RunStore

#: What this release says over the wire. See the module docstring.
PROTOCOL = 1
#: The header every request but ``/healthz`` names it in.
PROTOCOL_HEADER = "X-Strata-Protocol"


class ServiceError(Exception):
    """A request this host cannot honour."""


class RoundRequest(BaseModel):
    """One training round, as asked for from somewhere else."""

    #: A dataset in the catalog this host is configured against. Its version
    #: is already frozen — the caller made it — so two hosts materialising
    #: the same id get the same bytes.
    dataset_id: int
    #: What the caller means by that id: the name, version and digest its
    #: own catalog gave. Checked against this host's catalog, because a copy
    #: of a catalog keeps the identity and numbers its datasets on its own.
    #: See :func:`check_dataset`.
    dataset_name: str
    dataset_version: int
    annotation_digest: str | None
    #: A registered short name. File references are refused: see
    #: :func:`check_servable`.
    model: str
    params: dict = Field(default_factory=dict)
    #: Applied over ``params`` when the round starts cold. Both sets travel
    #: because only this host knows whether it has a parent: a caller asking
    #: for a warm start against a store that turns out to be empty would
    #: otherwise get a cold run trained for as long as a warm one.
    fresh_params: dict = Field(default_factory=dict)
    #: Cold start, ignoring whatever this host last trained on this dataset.
    fresh: bool = False
    #: Which catalog the caller believes this host serves. A dataset id is
    #: an integer meaningful only within one, so if the host is on another
    #: catalog the same id names different samples — and the round would
    #: succeed, silently, over the wrong data. Required: every catalog has
    #: an identity, and a client too old to send one is turned away by the
    #: protocol check before it gets here.
    catalog_id: str


class RoundResponse(BaseModel):
    run: Run
    metrics: dict[str, float] = Field(default_factory=dict)
    #: How the dataset was obtained, so a caller can tell a cache hit from a
    #: transfer without reading the host's logs.
    materialised: int = 0


class PredictionRequest(BaseModel):
    """Score some samples with a run this host holds.

    Checksums rather than paths: a path names a file on the caller's
    machine, and the whole point is that the caller has none. This host has
    the bucket and a cache, so it can turn content into files itself.
    """

    run_id: str
    checksums: list[str] = Field(default_factory=list)
    #: What the model is to be told about each sample, by checksum. Keyed
    #: rather than positional because the response is keyed too, and a
    #: caller that has to keep two lists aligned across a wire eventually
    #: does not.
    features: dict[str, dict] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    #: Checksum to the value the model produced. Keyed rather than ordered,
    #: because a sample the catalog does not know is simply absent and a
    #: positional answer could not say which.
    predictions: dict[str, AnyPrediction] = Field(default_factory=dict)
    #: Asked about but not in this catalog.
    unknown: list[str] = Field(default_factory=list)


QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


class Job(BaseModel):
    """A round this host has been asked to run, and how far it has got."""

    id: str
    state: str = QUEUED
    #: What the host is doing, in words, because a caller cannot see it.
    stage: str = "queued"
    done: int = 0
    total: int = 0
    result: RoundResponse | PredictionResponse | None = None
    #: The reason it failed, which is the only thing a caller can act on.
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.state in (DONE, FAILED)


class Jobs:
    """The rounds this host is running, and has run.

    In memory, and deliberately so: a job is a thread, and a thread does not
    survive a restart however carefully its record is written. A round that
    completed is in the run store, which is the durable half and the one
    worth recovering from.
    """

    def __init__(self, runner):
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, request, runner=None) -> Job:
        if isinstance(request, RoundRequest):
            check_servable(request.model)
        with self._lock:
            busy = next((j for j in self._jobs.values() if not j.finished), None)
            if busy is not None:
                raise BusyError(
                    f"Already running job {busy.id} ({busy.stage}). One round at a "
                    f"time: a second on the same GPU does not run slower, it runs "
                    f"out of memory and takes the first one with it."
                )
            job = Job(id=uuid.uuid4().hex[:12])
            self._jobs[job.id] = job

        work = runner or self._runner
        threading.Thread(target=self._work, args=(job, request, work), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _work(self, job: Job, request, runner) -> None:
        def report(stage: str, done: int = 0, total: int = 0) -> None:
            job.stage = stage
            job.done, job.total = done, total

        try:
            job.state = RUNNING
            job.stage = "starting"
            result = runner(request, report)
            # Set before done, or a caller that sees done first reads a job
            # with no result and cannot tell success from a lost one
            job.result = result
            job.stage = "finished"
            job.state = DONE
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"
            job.stage = "failed"
            job.state = FAILED


class BusyError(ServiceError):
    """A second round asked for while one is running."""


class CatalogMismatch(ServiceError):
    """A round prepared against one catalog, submitted to a host on another."""


class DatasetMismatch(ServiceError):
    """A dataset id that names other data here than where the round was prepared."""


def check_catalog(requested: str, serving: str) -> None:
    """Refuse a round prepared against a different catalog.

    ``dataset_id`` is an integer, and integers are only meaningful within
    one catalog: submitted to a host serving another, the same id names
    different samples and the round trains on the wrong data without
    failing.
    """
    if requested == serving:
        return
    raise CatalogMismatch(
        f"This host serves catalog {serving}, and the round was prepared "
        f"against {requested}. A dataset id names different samples in "
        f"each, so training it here would train on the wrong data."
    )


def check_dataset(request: RoundRequest, found) -> None:
    """Refuse a round whose dataset id names something else in this catalog.

    The catalog check is not enough alone. A copy of a catalog keeps its
    identity — that is what lets its answers be merged back — and numbers
    its datasets on its own, so dataset 12 on a laptop working from a copy
    and dataset 12 here can be different data with the catalog check
    passing. The round says what it means by the id; this checks that it
    means the same here. ``found`` is what this host's catalog says the id
    is, a :class:`~strata.catalog.DatasetRef`.
    """
    if (request.dataset_name, request.dataset_version) != (found.name, found.version):
        raise DatasetMismatch(
            f"Dataset {request.dataset_id} is {found.name} v{found.version} in this "
            f"host's catalog, and the round was prepared for {request.dataset_name} "
            f"v{request.dataset_version}. A copy of a catalog numbers its datasets on "
            f"its own, so the same id can name different data in each. Freeze the "
            f"dataset in the catalog this host reads, or merge the copy back first."
        )
    if request.annotation_digest != found.annotation_digest:
        raise DatasetMismatch(
            f"{found.name} v{found.version} has different answers in this host's "
            f"catalog than where the round was prepared — a copy labelled since it "
            f"was taken. Merge it back and freeze the dataset again."
        )


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
    report=None,
) -> RoundResponse:
    """Materialise a dataset version and train from it.

    ``datasets`` is where versions are written. Named by dataset and
    version, so two rounds over the same version share one directory rather
    than each fetching a copy.
    """
    from strata.catalog import ensure_materialised

    check_servable(request.model)

    check_catalog(request.catalog_id, catalog.id)
    check_dataset(request, catalog.dataset_named(request.dataset_id))

    def tick(done: int, total: int) -> None:
        if report is not None:
            report("materialising", done, total)

    # The same rule the laptop uses for when a version already on disk may
    # be reused, because it is the same function.
    result = ensure_materialised(
        catalog, request.dataset_id, datasets, on_progress=tick, cache=cache
    )
    manifest, target, materialised = result.manifest, result.directory, result.fetched

    previous = (
        None if request.fresh else store.latest(manifest.dataset, manifest.catalog_id)
    )
    params = (
        {**request.params, **request.fresh_params} if previous is None else request.params
    )

    # Said before training rather than after, because training is the long
    # part: a stage that only advances when a step finishes spends the whole
    # of the expensive step describing the cheap one that preceded it.
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
        ),
        store,
        on_epoch=epoch,
    )
    return RoundResponse(
        run=run, metrics=_metrics_of(store, run.id), materialised=materialised
    )


def run_prediction(
    request: PredictionRequest,
    catalog,
    store: RunStore,
    cache: Path,
    report=None,
) -> PredictionResponse:
    """Score samples by content, fetching whatever bytes are missing.

    The pool a review queue ranks is every unlabelled sample, so this is the
    expensive half of a push and the reason it belongs on the machine with
    the GPU rather than the machine with the reviewer.
    """
    from .handlers import predict as run_predict
    from .predictions import PredictionCache

    # Kept beside the runs, so the answer is the same for every caller
    # rather than for whichever machine asked first.
    known = PredictionCache.beside(store)
    # The host's cache is keyed on the same three inputs the caller's is:
    # a checkpoint, some bytes, and what the model was told. Computed here
    # from what arrived rather than sent, so the two sides cannot drift on
    # how a digest is taken.
    digests = {c: feature_digest(request.features.get(c, {})) for c in request.checksums}
    already = known.get(request.run_id, request.checksums, digests)
    wanted = [c for c in request.checksums if c not in already]

    def fetching(done: int, total: int) -> None:
        if report is not None:
            report("fetching", done, total)

    paths = catalog.ensure_cached(wanted, cache, on_progress=fetching)
    unknown = [c for c in wanted if c not in paths]

    made: dict[str, Prediction] = {}
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

    from strata.catalog import Catalog, CatalogError

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
        # Built by the same code the CLI uses, so the two cannot open a
        # bucket differently. Described from the environment until this
        # host reads a catalog file of its own.
        from strata.catalog.config import CatalogConfig, blobs_for

        endpoint = os.environ.get("STRATA_S3_ENDPOINT", "")
        if endpoint:
            bucket = _required("STRATA_S3_BUCKET", "An endpoint without a bucket names nothing.")
            local = None
        else:
            bucket = ""
            local = Path(_required(
                "STRATA_BLOBS_ROOT", "With no S3 endpoint the host reads files."
            ))
        return blobs_for(
            CatalogConfig(
                s3_endpoint=endpoint,
                s3_bucket=bucket,
                s3_region=os.environ.get("STRATA_S3_REGION", "garage"),
                s3_access_key=os.environ.get("STRATA_S3_ACCESS_KEY", ""),
                s3_secret_key=os.environ.get("STRATA_S3_SECRET_KEY", ""),
            ),
            local=local,
        )

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

    def served_catalog_id() -> str | None:
        # Memoised: a catalog's identity does not change under a running
        # host, and the alternative is a connection per submitted round
        # spent asking a question with one answer.
        if "id" not in served:
            served["id"] = catalog_for().id
        return served["id"]

    def run_one(request: RoundRequest, report) -> RoundResponse:
        return run_round(request, catalog_for(), store, datasets, cache, report=report)

    jobs = Jobs(run_one)
    app = FastAPI(title="strata modelling", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz() -> dict:
        # Unauthenticated, so a laptop can check it speaks this host's
        # protocol before sending anything at all
        return {"ok": True, "protocol": PROTOCOL}

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
                found = catalog_for().dataset_named(request.dataset_id)
            except CatalogError as e:
                raise ServiceError(str(e)) from None
            check_dataset(request, found)
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

        def run_it(req, report):
            return run_prediction(req, catalog_for(), store, cache, report)

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

        A caller cannot work this out for itself: run ids belong to the
        store that issued them, and the caller's own store is a different
        sequence naming different models.
        """
        run = store.latest(dataset, catalog)
        if run is None:
            raise HTTPException(status_code=404, detail=f"No runs over {dataset!r} here")
        return {"run": run.model_dump(mode="json"), "metrics": _metrics_of(store, run.id)}

    @app.get("/runs/{run_id}", dependencies=[Depends(authorise), Depends(spoken)])
    def get_run(run_id: str) -> dict:
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
