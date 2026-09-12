"""Training asked for from somewhere else.

The service is a shell: it materialises, picks a parent and calls the same
handler the in-process path calls. So what is worth testing is the shell —
what it refuses, and that the two paths agree.
"""

import threading
import time

import pytest
from pydantic import ValidationError

from strata.catalog import DatasetRef
from strata.modelling.service import (
    PROTOCOL,
    PROTOCOL_HEADER,
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

    def __init__(self, name, version, write, id="20260101T000000-aaaaaaaa"):
        self.id = id
        self.name = name
        self.version = version
        self._write = write
        self.materialised = 0

    @property
    def datasets(self):
        return self

    def named(self, dataset_id):
        return DatasetRef(self.name, self.version, None)

    def materialise(self, dataset_id, dest, on_progress=None, cache=None, features=None):
        self.materialised += 1
        self.features = list(features or [])
        self._write(dest)
        if on_progress is not None:
            on_progress(2, 2)
        return dest


CATALOG_ID = "20260101T000000-aaaaaaaa"


def _round(**overrides) -> RoundRequest:
    """A round as the laptop would ask for one, for the fake catalog's dataset."""
    fields = {
        "dataset_id": 1,
        "dataset_name": "d",
        "dataset_version": 2,
        "annotation_digest": None,
        "catalog_id": CATALOG_ID,
        "model": "stub",
    }
    fields.update(overrides)
    return RoundRequest(**fields)


def _refused(tmp_path, catalog, request: RoundRequest, match: str) -> None:
    """The round is refused for this reason, before anything is fetched."""
    from strata.modelling import RunStore

    with pytest.raises(ServiceError, match=match):
        run_round(request, catalog, RunStore.local(tmp_path / "runs"), tmp_path / "datasets")
    assert catalog.materialised == 0


@pytest.fixture
def fixture_dataset(tmp_path):
    """A materialised version, written the way the catalog would write it."""
    from strata.labels import (
        MANIFEST_FORMAT,
        MANIFEST_NAME,
        Choices,
        ClassificationSchema,
        Manifest,
        ManifestSample,
    )

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
                    split="val" if i >= 3 else "train",
                    value=Choices(values=["a"]),
                )
            )
        manifest = Manifest(
            format=MANIFEST_FORMAT,
            dataset="d",
            version=2,
            # The fake catalog's own, as the real one stamps every version
            catalog_id="20260101T000000-aaaaaaaa",
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
                id="",
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
        _round(),
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
        _round(),
        catalog,
        RunStore.local(tmp_path / "runs"),
        datasets,
    )

    assert catalog.materialised == 1
    assert (datasets / "d" / "v002" / "manifest.json").exists()
    assert result.metrics == {"val_accuracy": 0.5}


def test_a_manifest_from_before_formats_is_rebuilt(tmp_path, fixture_dataset, stub_training):
    """A copy this release cannot read is stale, not a reason to fail the round.

    Every host has such directories from before manifests said their format,
    and the catalog still holds each version they copied.
    """
    import json

    from strata.labels import MANIFEST_FORMAT
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    datasets = tmp_path / "datasets"
    stale = datasets / "d" / "v002"
    fixture_dataset(stale)
    written = json.loads((stale / "manifest.json").read_text())
    del written["format"]
    (stale / "manifest.json").write_text(json.dumps(written))

    run_round(
        _round(),
        catalog,
        RunStore.local(tmp_path / "runs"),
        datasets,
    )

    assert catalog.materialised == 1
    assert json.loads((stale / "manifest.json").read_text())["format"] == MANIFEST_FORMAT


def test_another_catalogs_folder_of_the_same_name_is_rebuilt(
    tmp_path, fixture_dataset, stub_training
):
    """The first round after pointing the host at a rebuilt catalog.

    Its numbering starts again, so its first version has the name and number
    of one this host already holds from the old catalog. Reusing that folder
    would train on the old data and report a number for the new.
    """
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset, id="20270101T000000-bbbbbbbb")
    datasets = tmp_path / "datasets"
    # Written by the old catalog, whose id the fixture stamps
    fixture_dataset(datasets / "d" / "v002")

    run_round(
        _round(catalog_id="20270101T000000-bbbbbbbb"),
        catalog,
        RunStore.local(tmp_path / "runs"),
        datasets,
    )

    assert catalog.materialised == 1


def test_the_parent_is_chosen_where_the_checkpoints_are(
    tmp_path, fixture_dataset, stub_training
):
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    store = RunStore.local(tmp_path / "runs")
    datasets = tmp_path / "datasets"

    first = run_round(_round(), catalog, store, datasets)
    assert stub_training["request"].parent_run_id is None

    # The caller never names a parent: only the host holding checkpoints can
    # pick one, or check that the one picked exists
    run_round(_round(), catalog, store, datasets)
    assert stub_training["request"].parent_run_id == first.run.id


def test_fresh_ignores_what_this_host_trained_before(
    tmp_path, fixture_dataset, stub_training
):
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    store = RunStore.local(tmp_path / "runs")
    datasets = tmp_path / "datasets"

    run_round(_round(), catalog, store, datasets)
    run_round(_round(fresh=True), catalog, store, datasets)
    assert stub_training["request"].parent_run_id is None


def test_a_file_reference_is_refused_before_anything_is_fetched(tmp_path, fixture_dataset):
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    store = RunStore.local(tmp_path / "runs")

    with pytest.raises(ServiceError):
        run_round(
            _round(model="model.py:Custom"),
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
    job = jobs.submit(_round())

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
    first = jobs.submit(_round())

    # Two training jobs on one GPU do not run slower, they run out of memory
    with pytest.raises(BusyError, match=first.id):
        jobs.submit(_round(dataset_id=2))
    release.set()


def test_a_failed_round_keeps_its_reason():
    from strata.modelling.service import Jobs

    def explode(request, progress):
        raise RuntimeError("CUDA out of memory")

    jobs = Jobs(explode)
    job = jobs.submit(_round())
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
        jobs.submit(_round(model="model.py:Custom"))


def test_the_stage_advances_past_materialising(tmp_path, fixture_dataset, stub_training):
    """The label has to follow the work, not the last thing that reported.

    Materialising reported progress and training reported nothing, so a job
    spent its whole ten-minute training run claiming to be fetching files —
    observable only by listening to the GPU.
    """
    from strata.modelling import RunStore

    stages = []
    run_round(
        _round(),
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
    job = jobs.submit(_round())
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
        PredictionRequest(run_id="1", checksums=list(known)),
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
        PredictionRequest(run_id="1", checksums=["a" * 64, "c" * 64]),
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
        PredictionRequest(run_id="1", checksums=["a" * 64]),
        FakeCache({"a" * 64: tmp_path / "a.jpg"}),
        RunStore.local(tmp_path / "runs"),
        tmp_path / "cache",
        report=lambda stage, done=0, total=0: stages.append(stage),
    )
    # Fetching a pool and scoring it are both minutes long, and a caller
    # watching from elsewhere cannot see either
    assert "fetching" in stages
    assert "predicting" in stages


def test_a_cold_round_takes_the_fresh_params_whatever_was_asked(
    tmp_path, fixture_dataset, stub_training
):
    """The host decides, because only the host knows if it has a parent.

    A caller asking for a warm start against a store that turns out to be
    empty would otherwise get a cold run trained on an increment's settings
    — undertrained, and indistinguishable from a real baseline afterwards.
    """
    from strata.modelling import RunStore

    catalog = FakeCatalog("d", 2, fixture_dataset)
    store = RunStore.local(tmp_path / "runs")
    datasets = tmp_path / "datasets"
    request = _round(
        params={"num_epochs": 4}, fresh_params={"num_epochs": 8}
    )

    run_round(request, catalog, store, datasets)
    assert stub_training["request"].params == {"num_epochs": 8}

    # And the second, which has something to continue, takes the increment's
    run_round(request, catalog, store, datasets)
    assert stub_training["request"].params == {"num_epochs": 4}


# ----------------------------------------------------------------------
# What a round must say, and what it is checked against
# ----------------------------------------------------------------------


def test_a_round_must_say_which_catalog_it_was_prepared_against():
    """Every catalog has an identity, so a round naming none is a client too old."""
    with pytest.raises(ValidationError, match="catalog_id"):
        RoundRequest(
            dataset_id=1,
            dataset_name="d",
            dataset_version=2,
            annotation_digest=None,
            model="stub",
        )


def test_a_round_for_another_catalog_is_refused_before_anything_is_fetched(
    tmp_path, fixture_dataset
):
    _refused(
        tmp_path,
        FakeCatalog("d", 2, fixture_dataset),
        _round(catalog_id="20250101T000000-cccccccc"),
        match="serves catalog",
    )


def test_an_id_that_names_another_dataset_here_is_refused(tmp_path, fixture_dataset):
    """A laptop working on a copy of the catalog.

    The copy keeps the catalog's identity, so the catalog check passes, and
    numbers its datasets on its own — so its dataset 1 and this host's
    dataset 1 can be different data.
    """
    _refused(
        tmp_path,
        FakeCatalog("d", 2, fixture_dataset),
        _round(dataset_name="other", dataset_version=5),
        match=r"d v2 .* other v5",
    )


def test_the_same_version_with_other_answers_is_refused(tmp_path, fixture_dataset):
    _refused(
        tmp_path,
        FakeCatalog("d", 2, fixture_dataset),
        _round(annotation_digest="f" * 64),
        match="different answers",
    )


# ----------------------------------------------------------------------
# Speaking the same protocol
# ----------------------------------------------------------------------


@pytest.fixture
def host(tmp_path, monkeypatch):
    """The service as it runs, over a real local catalog."""
    from fastapi.testclient import TestClient

    from strata.catalog import Catalog
    from strata.modelling.service import build

    catalog = Catalog.local(tmp_path / "catalog")
    config = tmp_path / "config.toml"
    config.write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    monkeypatch.setenv("STRATA_MODELLING_TOKEN", "t")
    monkeypatch.setenv("STRATA_CONFIG", str(config))
    monkeypatch.setenv("STRATA_MODELLING_ROOT", str(tmp_path / "modelling"))
    return TestClient(build()), catalog


def test_the_host_says_which_catalog_it_trains_from(host):
    client, catalog = host
    assert client.get("/healthz").json()["catalog"] == {"name": "default", "id": catalog.id}


SPOKEN = {"Authorization": "Bearer t", PROTOCOL_HEADER: str(PROTOCOL)}


def test_the_host_says_which_protocol_it_speaks(host):
    client, _ = host
    # Without a token: a laptop checks this before it sends anything at all
    assert client.get("/healthz").json()["protocol"] == PROTOCOL


def test_a_request_naming_no_protocol_is_refused(host):
    """A laptop from before protocols, which would not send what this host needs."""
    client, _ = host
    response = client.get("/models", headers={"Authorization": "Bearer t"})
    assert response.status_code == 426
    assert f"protocol {PROTOCOL}" in response.json()["detail"]


def test_a_request_in_this_protocol_is_served(host):
    client, _ = host
    assert client.get("/models", headers=SPOKEN).status_code == 200


def test_a_mismatched_dataset_is_refused_before_the_round_is_accepted(host, tmp_path):
    from strata.catalog import EVERYTHING
    from strata.labels import Choices, ClassificationSchema

    client, catalog = host
    paths = []
    for i in range(4):
        path = tmp_path / f"{i}.jpg"
        path.write_bytes(f"sample {i}".encode())
        paths.append(path)
    ids = catalog.ingest(paths, media="image")
    label_set = catalog.label_sets.create("x", ClassificationSchema(classes=["a"]))
    catalog.annotations.annotate_many(label_set, [(i, Choices(values=["a"])) for i in ids])
    dataset_id = catalog.create_dataset("d", label_set, collections=EVERYTHING)

    body = _round(
        dataset_id=dataset_id,
        dataset_name="not-d",
        dataset_version=1,
        catalog_id=catalog.id,
        model="multilabel",
    ).model_dump(mode="json")
    response = client.post("/round", json=body, headers=SPOKEN)

    # Refused, not accepted and then failed: a 202 for a round that cannot
    # run tells the caller nothing until it polls
    assert response.status_code == 400
    assert "not-d" in response.json()["detail"]


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------


def test_declared_features_reach_the_version_the_host_builds(
    tmp_path, fixture_dataset, stub_training
):
    """The bug this closes: the host materialised without them.

    A model that declared a feature it needed was refused; one that merely
    used features trained without them, and reported a number for it.
    """
    from strata.modelling import RunStore

    declared = {"name": "species", "source": "metadata", "ref": "species"}
    catalog = FakeCatalog("d", 2, fixture_dataset)
    run_round(
        _round(features=[declared]),
        catalog,
        RunStore.local(tmp_path / "runs"),
        tmp_path / "datasets",
    )

    assert [spec.as_dict() for spec in catalog.features] == [declared]


def test_a_feature_declaration_the_host_cannot_read_is_refused_first(
    tmp_path, fixture_dataset
):
    _refused(
        tmp_path,
        FakeCatalog("d", 2, fixture_dataset),
        _round(features=[{"name": "species", "ref": "species"}]),
        match="source",
    )
