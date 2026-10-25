"""Training asked for from somewhere else.

The service is a shell: it materialises, picks a parent and calls the same
handler the in-process path calls. So what is worth testing is the shell —
what it refuses, and that the two paths agree.
"""

import json
import threading
import time

import pytest
from pydantic import ValidationError

from strata.catalog import DatasetRef
from strata.modelling.remote.checks import check_servable
from strata.modelling.remote.rounds import run_round
from strata.modelling.remote.wire import PROTOCOL, PROTOCOL_HEADER, RoundRequest, ServiceError

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
    from strata.contracts import (
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
    import strata.modelling.remote.rounds as rounds
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

    monkeypatch.setattr(rounds, "run_train", fake)
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

    from strata.contracts import MANIFEST_FORMAT
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


def test_the_parent_is_chosen_where_the_checkpoints_are(tmp_path, fixture_dataset, stub_training):
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


def test_fresh_ignores_what_this_host_trained_before(tmp_path, fixture_dataset, stub_training):
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
    from strata.modelling.remote.jobs import Jobs

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
    from strata.modelling.remote.jobs import BusyError, Jobs

    release = threading.Event()
    jobs = Jobs(lambda request, progress: release.wait(2))
    first = jobs.submit(_round())

    # Two training jobs on one GPU do not run slower, they run out of memory
    with pytest.raises(BusyError, match=first.id):
        jobs.submit(_round(dataset_id=2))
    release.set()


def test_a_failed_round_keeps_its_reason():
    from strata.modelling.remote.jobs import Jobs

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
    from strata.modelling.remote.jobs import Jobs

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
    from strata.modelling.remote.jobs import Jobs

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

    Patched where run_prediction calls it by name.
    """
    from strata.contracts import ChoicesPrediction
    from strata.modelling.requests import ScoredPath

    seen = {}

    def fake(request, store, on_batch=None):
        seen["paths"] = list(request.paths)
        seen["checkpoint"] = request.checkpoint
        return [
            ScoredPath(path=path, value=ChoicesPrediction(values=["a"], confidences=[0.5]))
            for path in request.paths
        ]

    monkeypatch.setattr("strata.modelling.remote.rounds.run_predict", fake)
    return seen


def test_scoring_is_keyed_by_content(tmp_path, stub_predict):
    from strata.modelling import RunStore
    from strata.modelling.remote.rounds import run_prediction
    from strata.modelling.remote.wire import PredictionRequest

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
    from strata.modelling.remote.rounds import run_prediction
    from strata.modelling.remote.wire import PredictionRequest

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
    from strata.modelling.remote.rounds import run_prediction
    from strata.modelling.remote.wire import PredictionRequest

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


def test_scoring_names_the_checkpoint_it_was_asked_for(tmp_path, stub_predict):
    """And caches under it, so another checkpoint's answer is never handed back (0054)."""
    from strata.modelling import PredictionCache, RunStore
    from strata.modelling.remote.rounds import run_prediction
    from strata.modelling.remote.wire import PredictionRequest

    store = RunStore.local(tmp_path / "runs")
    catalog = FakeCache({"a" * 64: tmp_path / "a.jpg"})
    run_prediction(
        PredictionRequest(run_id="1", checkpoint="epoch-2", checksums=["a" * 64]),
        catalog,
        store,
        tmp_path / "cache",
    )

    assert stub_predict["checkpoint"] == "epoch-2"
    cache = PredictionCache.beside(store)
    assert set(cache.get("1", ["a" * 64], checkpoint="epoch-2")) == {"a" * 64}
    assert cache.get("1", ["a" * 64], checkpoint="epoch-1") == {}


def test_generating_here_fetches_the_inputs_samples_by_content(tmp_path, sft_run):
    from strata.modelling.remote.rounds import run_generation
    from strata.modelling.remote.wire import GenerationInputRef, GenerationRequest

    store, run = sft_run
    source = tmp_path / "source.txt"
    source.write_text("to summarise")
    catalog = FakeCache({"s" * 64: source})

    result = run_generation(
        GenerationRequest(
            run_id=run.id,
            checkpoint="epoch-1",
            inputs=[GenerationInputRef(values={"subject": "hi"}, samples={"source": ["s" * 64]})],
            settings=[{"temperature": 0.7}],
            seeds=[4],
        ),
        catalog,
        store,
        tmp_path / "cache",
    )

    [out] = result.outputs
    made = json.loads(out.data)
    assert made["samples"] == {"source": ["to summarise"]}
    assert (made["epoch"], out.checkpoint, out.seed) == (1, "epoch-1", 4)


def test_generating_with_a_sample_the_host_does_not_know_makes_nothing(tmp_path, sft_run):
    from strata.modelling.remote.rounds import run_generation
    from strata.modelling.remote.wire import GenerationInputRef, GenerationRequest

    store, run = sft_run
    with pytest.raises(ServiceError, match="does not know 1 of the samples"):
        run_generation(
            GenerationRequest(
                run_id=run.id,
                inputs=[GenerationInputRef(samples={"source": ["u" * 64]})],
                settings=[{}],
                seeds=[0],
            ),
            FakeCache({}),
            store,
            tmp_path / "cache",
        )


def test_generated_bytes_survive_the_wire(tmp_path, sft_run):
    """Base64 rather than text, so an image would come back as it went (0054)."""
    from strata.modelling.remote.jobs import Job
    from strata.modelling.remote.wire import GenerationResponse
    from strata.modelling.requests import GeneratedOutput

    raw = bytes(range(256))
    job = Job(
        id="j",
        result=GenerationResponse(
            outputs=[GeneratedOutput(data=raw, settings_digest="d", seed=1, checkpoint="final")]
        ),
    )

    sent = json.loads(job.model_dump_json())
    assert GenerationResponse.model_validate(sent["result"]).outputs[0].data == raw


@pytest.fixture
def sft_run(tmp_path):
    """A media model's run with three checkpoints, recorded here."""
    from media_model import ECHO, ECHO_MODEL

    from strata.contracts import MANIFEST_FORMAT
    from strata.modelling import RunStore, TrainRequest, train

    root = tmp_path / "sft"
    (root / "files").mkdir(parents=True)
    (root / "echo.py").write_text(ECHO_MODEL)
    samples = []
    for i in range(3):
        (root / "files" / f"{i}.txt").write_text(f"body {i}")
        samples.append(
            {
                "checksum": f"{i:064d}",
                "type": "text/sft",
                "path": f"files/{i}.txt",
                "split": "train",
                "value": {"kind": "choices", "values": ["keep"]},
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "dataset": "sft",
                "label_set": "verdicts",
                "label_schema": {"label_type": "classification", "classes": ["keep"]},
                "samples": samples,
            }
        )
    )
    store = RunStore.local(tmp_path / "runs")
    return store, train(TrainRequest(dataset_dir=root, model=ECHO), store)


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
    request = _round(params={"num_epochs": 4}, fresh_params={"num_epochs": 8})

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
# The split the caller holds


def _held(fixture_dataset, tmp_path):
    """The version as the caller's directory has it, for sending its split."""
    from strata.contracts import MANIFEST_NAME, Manifest

    where = tmp_path / "held"
    fixture_dataset(where)
    return Manifest.model_validate_json((where / MANIFEST_NAME).read_text())


def test_a_split_the_caller_drew_is_applied_to_a_copy(tmp_path, fixture_dataset, stub_training):
    from strata.contracts import MANIFEST_NAME, Manifest, order_digest, sides_string
    from strata.modelling import RunStore
    from strata.modelling.remote.wire import SplitSides

    held = _held(fixture_dataset, tmp_path)
    # Every side flipped, so the draw differs from the version's everywhere
    drawn = "".join("h" if c != "h" else "t" for c in sides_string(held))
    split = SplitSides(sides=drawn, order_digest=order_digest(held), holdout_ratio=0.5)
    catalog = FakeCatalog("d", 2, fixture_dataset)
    datasets = tmp_path / "datasets"

    run_round(_round(split=split), catalog, RunStore.local(tmp_path / "runs"), datasets)

    trained = stub_training["request"].dataset_dir
    assert trained != datasets / "d" / "v002" and trained.parent == datasets / "d"
    copy = Manifest.model_validate_json((trained / MANIFEST_NAME).read_text())
    assert sides_string(copy) == drawn
    assert copy.holdout_ratio == 0.5
    assert copy.holdout_ratio_achieved == drawn.count("h") / len(drawn)
    # The version this host materialised keeps the catalog's sides
    original = Manifest.model_validate_json((datasets / "d" / "v002" / MANIFEST_NAME).read_text())
    assert sides_string(original) == sides_string(held)


def test_a_split_equal_to_the_versions_trains_from_the_version(
    tmp_path, fixture_dataset, stub_training
):
    from strata.contracts import order_digest, sides_string
    from strata.modelling import RunStore
    from strata.modelling.remote.wire import SplitSides

    held = _held(fixture_dataset, tmp_path)
    split = SplitSides(sides=sides_string(held), order_digest=order_digest(held))
    datasets = tmp_path / "datasets"

    run_round(
        _round(split=split),
        FakeCatalog("d", 2, fixture_dataset),
        RunStore.local(tmp_path / "runs"),
        datasets,
    )

    assert stub_training["request"].dataset_dir == datasets / "d" / "v002"


def test_a_split_over_other_samples_is_refused_before_training(
    tmp_path, fixture_dataset, stub_training
):
    from strata.contracts import sides_string
    from strata.modelling import RunStore
    from strata.modelling.remote.checks import SplitMismatch
    from strata.modelling.remote.wire import SplitSides

    held = _held(fixture_dataset, tmp_path)
    split = SplitSides(sides=sides_string(held), order_digest="0" * 64)

    with pytest.raises(SplitMismatch, match="order"):
        run_round(
            _round(split=split),
            FakeCatalog("d", 2, fixture_dataset),
            RunStore.local(tmp_path / "runs"),
            tmp_path / "datasets",
        )
    assert "request" not in stub_training


def test_a_split_of_the_wrong_length_is_refused(tmp_path, fixture_dataset, stub_training):
    from strata.contracts import order_digest
    from strata.modelling import RunStore
    from strata.modelling.remote.checks import SplitMismatch
    from strata.modelling.remote.wire import SplitSides

    held = _held(fixture_dataset, tmp_path)
    split = SplitSides(sides="t", order_digest=order_digest(held))

    with pytest.raises(SplitMismatch, match="names 1 sample"):
        run_round(
            _round(split=split),
            FakeCatalog("d", 2, fixture_dataset),
            RunStore.local(tmp_path / "runs"),
            tmp_path / "datasets",
        )
    assert "request" not in stub_training


# Speaking the same protocol
# ----------------------------------------------------------------------


@pytest.fixture
def host(tmp_path, monkeypatch):
    """The service as it runs, over a real local catalog."""
    from fastapi.testclient import TestClient

    from strata.catalog import Catalog
    from strata.modelling.remote.service import build

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


def test_the_host_generates_from_a_run_it_holds(tmp_path, monkeypatch, sft_run):
    """Accepted as a job on the same queue, and answered with every output (0054)."""
    from fastapi.testclient import TestClient

    from strata.catalog import Catalog
    from strata.modelling.remote.service import build

    _, run = sft_run
    Catalog.local(tmp_path / "catalog")
    config = tmp_path / "config.toml"
    config.write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    monkeypatch.setenv("STRATA_MODELLING_TOKEN", "t")
    monkeypatch.setenv("STRATA_CONFIG", str(config))
    # The host's store is the one the fixture trained into
    monkeypatch.setenv("STRATA_MODELLING_ROOT", str(tmp_path))
    client = TestClient(build())

    accepted = client.post(
        "/generate",
        headers=SPOKEN,
        json={
            "run_id": run.id,
            "inputs": [{"values": {"subject": "hi"}}],
            "settings": [{}],
            "seeds": [2],
        },
    )
    assert accepted.status_code == 202
    for _ in range(100):
        job = client.get(f"/jobs/{accepted.json()['id']}", headers=SPOKEN).json()
        if job["state"] in ("done", "failed"):
            break
        time.sleep(0.02)

    assert job["state"] == "done", job.get("error")
    [out] = job["result"]["outputs"]
    assert (out["checkpoint"], out["seed"]) == ("epoch-3", 2)


def test_a_mismatched_dataset_is_refused_before_the_round_is_accepted(host, tmp_path):
    from strata.catalog import EVERYTHING
    from strata.contracts import Choices, ClassificationSchema

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


def test_a_feature_declaration_the_host_cannot_read_is_refused_first(tmp_path, fixture_dataset):
    _refused(
        tmp_path,
        FakeCatalog("d", 2, fixture_dataset),
        _round(features=[{"name": "species", "ref": "species"}]),
        match="source",
    )
