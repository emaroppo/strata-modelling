"""Training asked for from somewhere else.

The service is a shell: it materialises, picks a parent and calls the same
handler the in-process path calls. So what is worth testing is the shell —
what it refuses, and that the two paths agree.
"""

import threading
import time

import pytest

from strata.modelling.service import (
    RoundRequest,
    ServiceError,
    check_servable,
    run_round,
)

# ----------------------------------------------------------------------
# What it will not serve
# ----------------------------------------------------------------------


def test_a_registry_name_is_servable():
    assert check_servable("multilabel") is None


@pytest.mark.parametrize(
    "ref",
    [
        "/home/someone/projects/thing/model.py:Custom",
        "model.py:Custom",
        "some.module:Custom",
    ],
)
def test_a_direct_reference_is_refused(ref):
    # It names code on the caller's machine. Importing it here fails
    # confusingly at best, and at worst finds a different file of the same
    # name — a service that imports whatever path it is handed is not
    # serving, it is executing.
    with pytest.raises(ServiceError) as caught:
        check_servable(ref)
    message = str(caught.value)
    # Both real answers, because a resolution failure implies neither and
    # the local one is easy to forget is still available
    assert "entry point" in message
    assert "locally" in message


def test_the_refusal_lists_what_this_host_has():
    with pytest.raises(ServiceError, match="multilabel"):
        check_servable("model.py:Custom")


# ----------------------------------------------------------------------
# The round
# ----------------------------------------------------------------------


class FakeCatalog:
    """A catalog that knows one dataset and writes it out on request."""

    def __init__(self, name, version, write):
        self.name = name
        self.version = version
        self._write = write
        self.materialised = 0

    def dataset_named(self, dataset_id):
        return self.name, self.version

    def materialise(self, dataset_id, dest, on_progress=None, cache=None):
        self.materialised += 1
        self._write(dest)
        if on_progress is not None:
            on_progress(2, 2)
        return dest


@pytest.fixture
def fixture_dataset(tmp_path):
    """A materialised version, written the way the catalog would write it."""
    from strata.catalog import MANIFEST_NAME, Manifest, ManifestSample
    from strata.labels import Choices, ClassificationSchema

    def write(dest):
        files = dest / "files"
        files.mkdir(parents=True, exist_ok=True)
        samples = []
        for i in range(4):
            name = f"{i:064x}.txt"
            (files / name).write_bytes(f"sample {i}".encode())
            samples.append(
                ManifestSample(
                    id=i,
                    checksum=f"{i:064x}",
                    path=f"files/{name}",
                    val=i >= 3,
                    value=Choices(values=["a"]),
                )
            )
        manifest = Manifest(
            dataset="d",
            version=2,
            label_set="x",
            label_schema=ClassificationSchema(classes=["a"]),
            samples=samples,
        )
        (dest / MANIFEST_NAME).write_text(manifest.model_dump_json())

    return write


@pytest.fixture
def stub_training(monkeypatch):
    """Stand in for the handler, and record what it was handed.

    The handler itself is covered where it lives; what matters here is the
    request the shell builds — which directory, which parent.
    """
    import strata.modelling.service as service
    from strata.modelling.requests import Run

    seen = {}

    def fake(request, store, on_epoch=None):
        seen["request"] = request
        return store.record(
            Run(
                id=0,
                parent_run_id=request.parent_run_id,
                dataset="d",
                dataset_version=2,
                label_set="x",
                model=request.model,
                model_version="1",
                params=request.params,
                classes=["a"],
            ),
            {"val_accuracy": 0.5},
        )

    monkeypatch.setattr(service, "run_train", fake)
    return seen


def test_a_version_already_present_is_not_materialised_again(
    tmp_path, fixture_dataset, stub_training
):
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    datasets = tmp_path / "datasets"
    # Written where the service would have put it
    fixture_dataset(datasets / "d" / "v002")

    result = run_round(
        RoundRequest(dataset_id=1, model="stub"),
        catalog,
        RunStore.local(tmp_path / "runs"),
        datasets,
    )

    assert catalog.materialised == 0
    assert result.materialised == 0
    assert stub_training["request"].dataset_dir == datasets / "d" / "v002"


def test_a_missing_version_is_materialised(tmp_path, fixture_dataset, stub_training):
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    datasets = tmp_path / "datasets"

    result = run_round(
        RoundRequest(dataset_id=1, model="stub"),
        catalog,
        RunStore.local(tmp_path / "runs"),
        datasets,
    )

    assert catalog.materialised == 1
    assert (datasets / "d" / "v002" / "manifest.json").exists()
    assert result.metrics == {"val_accuracy": 0.5}


def test_the_parent_is_chosen_where_the_checkpoints_are(
    tmp_path, fixture_dataset, stub_training
):
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    store = RunStore.local(tmp_path / "runs")
    datasets = tmp_path / "datasets"

    first = run_round(RoundRequest(dataset_id=1, model="stub"), catalog, store, datasets)
    assert stub_training["request"].parent_run_id is None

    # The caller never names a parent: only the host holding checkpoints can
    # pick one, or check that the one picked exists
    run_round(RoundRequest(dataset_id=1, model="stub"), catalog, store, datasets)
    assert stub_training["request"].parent_run_id == first.run.id


def test_fresh_ignores_what_this_host_trained_before(
    tmp_path, fixture_dataset, stub_training
):
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    store = RunStore.local(tmp_path / "runs")
    datasets = tmp_path / "datasets"

    run_round(RoundRequest(dataset_id=1, model="stub"), catalog, store, datasets)
    run_round(RoundRequest(dataset_id=1, model="stub", fresh=True), catalog, store, datasets)
    assert stub_training["request"].parent_run_id is None


def test_a_file_reference_is_refused_before_anything_is_fetched(tmp_path, fixture_dataset):
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    store = RunStore.local(tmp_path / "runs")

    with pytest.raises(ServiceError):
        run_round(
            RoundRequest(dataset_id=1, model="model.py:Custom"),
            catalog,
            store,
            tmp_path / "datasets",
        )
    # Refusing after a transfer would waste the expensive part on a request
    # that was never going to work
    assert catalog.materialised == 0


# ----------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------


def test_a_round_is_accepted_and_then_run():
    from strata.modelling.service import Jobs

    started = threading.Event()
    release = threading.Event()

    def runner(request, progress):
        started.set()
        release.wait(2)
        progress(4, 4)
        return "done-ish"

    jobs = Jobs(runner)
    job = jobs.submit(RoundRequest(dataset_id=1, model="stub"))

    # The point of submitting: this returned before the work did
    assert started.wait(2)
    assert jobs.get(job.id).state in ("queued", "running")

    release.set()
    for _ in range(100):
        if jobs.get(job.id).finished:
            break
        time.sleep(0.02)
    assert jobs.get(job.id).state == "done"
    assert jobs.get(job.id).result == "done-ish"


def test_a_second_round_is_refused_while_one_runs():
    from strata.modelling.service import BusyError, Jobs

    release = threading.Event()
    jobs = Jobs(lambda request, progress: release.wait(2))
    first = jobs.submit(RoundRequest(dataset_id=1, model="stub"))

    # Two training jobs on one GPU do not run slower, they run out of memory
    with pytest.raises(BusyError, match=first.id):
        jobs.submit(RoundRequest(dataset_id=2, model="stub"))
    release.set()


def test_a_failed_round_keeps_its_reason():
    from strata.modelling.service import Jobs

    def explode(request, progress):
        raise RuntimeError("CUDA out of memory")

    jobs = Jobs(explode)
    job = jobs.submit(RoundRequest(dataset_id=1, model="stub"))
    for _ in range(100):
        if jobs.get(job.id).finished:
            break
        time.sleep(0.02)

    # The only thing a caller can act on, so it has to survive the thread
    assert jobs.get(job.id).state == "failed"
    assert "CUDA out of memory" in jobs.get(job.id).error


def test_an_unservable_model_is_refused_at_submission():
    from strata.modelling.service import Jobs

    jobs = Jobs(lambda request, progress: None)
    # Before a thread starts, so a caller learns immediately rather than by
    # polling a job that was never going to run
    with pytest.raises(ServiceError, match="entry point"):
        jobs.submit(RoundRequest(dataset_id=1, model="model.py:Custom"))


def test_the_stage_advances_past_materialising(tmp_path, fixture_dataset, stub_training):
    """The label has to follow the work, not the last thing that reported.

    Materialising reported progress and training reported nothing, so a job
    spent its whole ten-minute training run claiming to be fetching files —
    observable only by listening to the GPU.
    """
    from strata.modelling import RunStore

    stages = []
    run_round(
        RoundRequest(dataset_id=1, model="stub"),
        FakeCatalog("d", 2, fixture_dataset),
        RunStore.local(tmp_path / "runs"),
        tmp_path / "datasets",
        report=lambda stage, done=0, total=0: stages.append(stage),
    )

    assert "materialising" in stages
    assert stages[-1] == "training"


def test_a_job_says_what_it_is_doing():
    from strata.modelling.service import Jobs

    seen = []
    release = threading.Event()

    def runner(request, report):
        report("materialising", 1, 2)
        seen.append("materialising")
        report("training")
        seen.append("training")
        release.wait(2)
        return "result"

    jobs = Jobs(runner)
    job = jobs.submit(RoundRequest(dataset_id=1, model="stub"))
    for _ in range(100):
        if jobs.get(job.id).stage == "training":
            break
        time.sleep(0.02)

    assert jobs.get(job.id).stage == "training"
    release.set()


# ----------------------------------------------------------------------
# Scoring a pool
# ----------------------------------------------------------------------


class FakeCache:
    """A catalog that can produce files for checksums it knows."""

    def __init__(self, known):
        self.known = known
        self.asked = []

    def ensure_cached(self, checksums, cache, on_progress=None):
        self.asked.append(list(checksums))
        found = {c: self.known[c] for c in checksums if c in self.known}
        if on_progress is not None:
            on_progress(len(found), len(found))
        return found


@pytest.fixture
def stub_predict(monkeypatch):
    """Stand in for inference; the handler is covered where it lives.

    run_prediction imports the handler when it runs, so patching the
    handler's own module is what takes effect.
    """
    from strata.labels import ChoicesPrediction
    from strata.modelling.requests import ScoredPath

    seen = {}

    def fake(request, store, on_batch=None):
        seen["paths"] = list(request.paths)
        return [
            ScoredPath(path=path, value=ChoicesPrediction(values=["a"], confidences=[0.5]))
            for path in request.paths
        ]

    monkeypatch.setattr("strata.modelling.handlers.predict", fake)
    return seen


def test_scoring_is_keyed_by_content(tmp_path, stub_predict):
    from strata.modelling import RunStore
    from strata.modelling.service import PredictionRequest, run_prediction

    known = {"a" * 64: tmp_path / "a.jpg", "b" * 64: tmp_path / "b.jpg"}
    catalog = FakeCache(known)

    result = run_prediction(
        PredictionRequest(run_id=1, checksums=list(known)),
        catalog,
        RunStore.local(tmp_path / "runs"),
        tmp_path / "cache",
    )

    # Keyed rather than positional: a sample the catalog does not know is
    # absent, and a positional answer could not say which one
    assert set(result.predictions) == set(known)
    assert result.unknown == []


def test_a_sample_the_host_does_not_know_is_reported(tmp_path, stub_predict):
    from strata.modelling import RunStore
    from strata.modelling.service import PredictionRequest, run_prediction

    catalog = FakeCache({"a" * 64: tmp_path / "a.jpg"})
    result = run_prediction(
        PredictionRequest(run_id=1, checksums=["a" * 64, "c" * 64]),
        catalog,
        RunStore.local(tmp_path / "runs"),
        tmp_path / "cache",
    )
    assert result.unknown == ["c" * 64]
    assert set(result.predictions) == {"a" * 64}


def test_scoring_says_what_it_is_doing(tmp_path, stub_predict):
    from strata.modelling import RunStore
    from strata.modelling.service import PredictionRequest, run_prediction

    stages = []
    run_prediction(
        PredictionRequest(run_id=1, checksums=["a" * 64]),
        FakeCache({"a" * 64: tmp_path / "a.jpg"}),
        RunStore.local(tmp_path / "runs"),
        tmp_path / "cache",
        report=lambda stage, done=0, total=0: stages.append(stage),
    )
    # Fetching a pool and scoring it are both minutes long, and a caller
    # watching from elsewhere cannot see either
    assert "fetching" in stages
    assert "predicting" in stages
