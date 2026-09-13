"""Modelling's stages: training here or there, and scoring by one implementation."""

import json

import pytest
from counting_model import COUNTER

from strata.labels import MANIFEST_NAME
from strata.modelling.remote.checks import CatalogMismatch
from strata.modelling.remote.client import RemoteError
from strata.modelling.stages import (
    Context,
    DatasetIdentity,
    EvaluateRequest,
    Host,
    StageError,
    TrainStageRequest,
    evaluate,
    train,
)


def _train(directory, **overrides) -> TrainStageRequest:
    return TrainStageRequest(**{"dataset_dir": directory, "model": COUNTER, **overrides})


# ----------------------------------------------------------------------
# train, here
# ----------------------------------------------------------------------


def test_a_cold_round_records_a_run_with_the_cold_parameters(store, dataset_dir):
    record = train(
        _train(dataset_dir(), params={"bias": 0.4}, fresh_params={"bias": 0.9}), Context(store)
    )
    run = store.get(record.run_id)
    assert record.where == "local" and record.parent_run_id is None
    assert run.params == {"bias": 0.9}
    assert record.metrics["accuracy"] == pytest.approx(0.9)
    assert record.checkpoint.exists()


def test_the_default_policy_continues_from_the_newest_run(store, dataset_dir):
    directory = dataset_dir()
    request = _train(directory, params={"bias": 0.4}, fresh_params={"bias": 0.9})
    first = train(request, Context(store))
    second = train(request, Context(store))
    assert second.parent_run_id == first.run_id
    # Warm: the fresh parameters are not applied
    assert store.get(second.run_id).params == {"bias": 0.4}


def test_a_round_after_a_re_split_starts_cold(store, dataset_dir):
    # Version 3 drew its sides from nothing; the run over version 1 may have
    # trained on what 3 now holds out, so it is not continued
    before = train(_train(dataset_dir(version=1, sides_from_version=1)), Context(store))
    after = train(_train(dataset_dir(version=3, sides_from_version=3)), Context(store))
    assert after.parent_run_id is None
    # And from there the lineage continues as usual
    again = train(_train(dataset_dir(version=4, sides_from_version=3)), Context(store))
    assert again.parent_run_id == after.run_id != before.run_id


def test_fresh_starts_cold_whatever_the_store_holds(store, dataset_dir):
    directory = dataset_dir()
    train(_train(directory), Context(store))
    record = train(_train(directory, fresh=True), Context(store))
    assert record.parent_run_id is None


def test_a_named_parent_is_the_one_continued_from(store, dataset_dir):
    directory = dataset_dir()
    first = train(_train(directory), Context(store))
    train(_train(directory), Context(store))
    record = train(_train(directory, parent=first.run_id), Context(store))
    assert record.parent_run_id == first.run_id


def test_a_parent_the_store_does_not_hold_is_refused(store, dataset_dir):
    with pytest.raises(StageError, match="No run 'nope'"):
        train(_train(dataset_dir(), parent="nope"), Context(store))


def test_training_here_needs_a_directory(store):
    with pytest.raises(StageError, match="materialised directory"):
        train(TrainStageRequest(model=COUNTER), Context(store))


# ----------------------------------------------------------------------
# train, there
# ----------------------------------------------------------------------


class FakeHost:
    """The client's surface, answering as a host would."""

    served = {"name": "main", "id": "cat-1"}
    outcome = {"state": "done"}

    def __init__(self, url, token):
        self.url, self.token = url, token
        self.submitted = []

    def served_catalog(self):
        return self.served

    def submit(self, request):
        self.submitted.append(request)
        return {"id": "job-1"}

    def follow(self, job_id, on_state=None):
        if on_state:
            on_state({"state": "running", "stage": "training"})
        run = {
            "id": "20260101T000000000000-gpu",
            "dataset": "d",
            "dataset_version": 1,
            "label_set": "presence",
            "model": "multilabel",
            "model_version": "1",
            "classes": ["cat", "dog"],
            "metrics": {"val_accuracy": 0.8},
        }
        return {
            **self.outcome,
            "result": {"run": run, "metrics": {"val_accuracy": 0.8}, "materialised": 12},
        }


def _identity(**overrides) -> DatasetIdentity:
    fields = {
        "dataset_id": 7,
        "name": "d",
        "version": 1,
        "annotation_digest": "a" * 64,
        "catalog_id": "cat-1",
    }
    return DatasetIdentity(**{**fields, **overrides})


def _there(**overrides) -> Context:
    return Context(store=None, host=Host("http://gpu", "t"), client=FakeHost, **overrides)


def test_a_remote_round_sends_the_identity_and_records_the_hosts_run():
    states = []
    record = train(
        TrainStageRequest(
            dataset=_identity(),
            model="multilabel",
            params={"lr": 1e-5},
            fresh=True,
            features=[{"name": "species", "source": "metadata", "ref": "species"}],
        ),
        _there(on_state=states.append),
    )
    assert record.where == "remote"
    assert record.run_id.endswith("-gpu")
    assert record.metrics == {"val_accuracy": 0.8} and record.materialised == 12
    assert states == [{"state": "running", "stage": "training"}]


def test_what_is_sent_is_the_hosts_own_request(monkeypatch):
    sent = []
    monkeypatch.setattr(FakeHost, "submit", lambda self, r: sent.append(r) or {"id": "job"})
    train(TrainStageRequest(dataset=_identity(), model="multilabel", fresh=True), _there())
    [request] = sent
    assert (request.dataset_id, request.dataset_name, request.dataset_version) == (7, "d", 1)
    assert request.catalog_id == "cat-1" and request.fresh is True


def test_the_split_this_side_holds_is_sent_positionally(monkeypatch, dataset_dir):
    from strata.labels import MANIFEST_NAME, Manifest, order_digest, sides_string

    sent = []
    monkeypatch.setattr(FakeHost, "submit", lambda self, r: sent.append(r) or {"id": "job"})
    directory = dataset_dir(n_train=3, n_val=1)
    request = TrainStageRequest(
        dataset_dir=directory, dataset=_identity(), model="multilabel", fresh=True
    )
    train(request, _there())
    [request] = sent
    held = Manifest.model_validate_json((directory / MANIFEST_NAME).read_text())
    assert request.split.sides == sides_string(held)
    assert sorted(request.split.sides) == ["t", "t", "t", "v"]
    assert request.split.order_digest == order_digest(held)


def test_a_round_with_no_directory_sends_no_split(monkeypatch):
    sent = []
    monkeypatch.setattr(FakeHost, "submit", lambda self, r: sent.append(r) or {"id": "job"})
    train(TrainStageRequest(dataset=_identity(), model="multilabel", fresh=True), _there())
    assert sent[0].split is None


def test_a_host_on_another_catalog_is_refused_before_anything_is_sent(monkeypatch):
    monkeypatch.setattr(FakeHost, "served", {"name": "main", "id": "other"})
    with pytest.raises(CatalogMismatch, match="serves catalog other"):
        train(TrainStageRequest(dataset=_identity(), model="multilabel"), _there())


def test_a_failed_job_is_an_error_naming_the_hosts_reason(monkeypatch):
    monkeypatch.setattr(FakeHost, "outcome", {"state": "failed", "error": "OOM"})
    with pytest.raises(RemoteError, match="OOM"):
        train(TrainStageRequest(dataset=_identity(), model="multilabel"), _there())


def test_a_remote_round_needs_the_datasets_identity():
    with pytest.raises(StageError, match="identity"):
        train(TrainStageRequest(model="multilabel"), _there())


# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------


def _hold_out(directory, n: int) -> None:
    """Move the last n training samples to the holdout, in the manifest."""
    path = directory / MANIFEST_NAME
    payload = json.loads(path.read_text())
    moved = 0
    for sample in reversed(payload["samples"]):
        if sample["split"] == "train" and moved < n:
            sample["split"] = "holdout"
            moved += 1
    path.write_text(json.dumps(payload))


def test_evaluate_scores_one_side_by_one_implementation(store, dataset_dir):
    directory = dataset_dir(n_train=8, n_val=2)
    _hold_out(directory, 3)
    record = train(_train(directory, fresh=True), Context(store))

    scored = evaluate(EvaluateRequest(run_id=record.run_id, dataset_dir=directory), Context(store))

    # The counting model predicts the first class, which is every answer
    assert (scored.side, scored.samples, scored.where) == ("holdout", 3, "local")
    assert scored.metrics == {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert scored.per_class["cat"].support == 3 and scored.per_class["cat"].f1 == 1.0


def test_evaluate_can_score_validation_too(store, dataset_dir):
    directory = dataset_dir(n_train=8, n_val=2)
    record = train(_train(directory, fresh=True), Context(store))
    scored = evaluate(
        EvaluateRequest(run_id=record.run_id, dataset_dir=directory, side="val"), Context(store)
    )
    assert scored.samples == 2


def test_a_scored_sample_is_not_scored_twice(store, dataset_dir, monkeypatch):
    directory = dataset_dir(n_train=8, n_val=2)
    _hold_out(directory, 3)
    record = train(_train(directory, fresh=True), Context(store))
    request = EvaluateRequest(run_id=record.run_id, dataset_dir=directory)
    first = evaluate(request, Context(store))

    import strata.modelling.handlers as handlers

    monkeypatch.setattr(handlers, "predict", lambda *a, **k: pytest.fail("predicted again"))
    assert evaluate(request, Context(store)).metrics == first.metrics


def test_a_wrong_answer_is_counted(store, dataset_dir):
    directory = dataset_dir(n_train=8, n_val=2, classes=("cat", "dog"))
    _hold_out(directory, 4)
    record = train(_train(directory, fresh=True), Context(store))
    # Flip one held-out answer to the class the model never predicts
    path = directory / MANIFEST_NAME
    payload = json.loads(path.read_text())
    held = [s for s in payload["samples"] if s["split"] == "holdout"]
    held[0]["value"] = {"kind": "choices", "values": ["dog"]}
    path.write_text(json.dumps(payload))

    scored = evaluate(EvaluateRequest(run_id=record.run_id, dataset_dir=directory), Context(store))
    assert scored.metrics["exact_match"] == pytest.approx(0.75)
    assert scored.metrics["precision"] == pytest.approx(0.75)
    assert scored.metrics["recall"] == pytest.approx(0.75)
    # Per class: cat was asserted on three and guessed on four; dog asserted
    # on one and never guessed
    assert scored.per_class["cat"].precision == pytest.approx(0.75)
    assert scored.per_class["cat"].recall == 1.0
    assert scored.per_class["dog"] .recall == 0.0 and scored.per_class["dog"].support == 1


def test_an_empty_side_is_refused(store, dataset_dir):
    directory = dataset_dir(n_train=8, n_val=2)
    record = train(_train(directory, fresh=True), Context(store))
    with pytest.raises(StageError, match="no answered sample on the 'holdout' side"):
        evaluate(EvaluateRequest(run_id=record.run_id, dataset_dir=directory), Context(store))


def test_only_classification_is_scored_for_now(store, dataset_dir):
    directory = dataset_dir(n_train=8, n_val=2)
    _hold_out(directory, 2)
    record = train(_train(directory, fresh=True), Context(store))
    path = directory / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["label_schema"] = {"task": "span", "classes": ["name"]}
    path.write_text(json.dumps(payload))
    with pytest.raises(StageError, match="classification only"):
        evaluate(EvaluateRequest(run_id=record.run_id, dataset_dir=directory), Context(store))


def test_evaluate_asks_the_host_when_there_is_one(dataset_dir):
    directory = dataset_dir(n_train=8, n_val=2)
    _hold_out(directory, 2)
    payload = json.loads((directory / MANIFEST_NAME).read_text())
    held = [s["checksum"] for s in payload["samples"] if s["split"] == "holdout"]

    class Scores(FakeHost):
        def predict(self, request):
            self.asked = request
            return {"id": "job-2"}

        def follow(self, job_id, on_state=None):
            cat = {"kind": "choices", "values": ["cat"], "confidences": [0.9]}
            answers = {c: cat for c in held}
            return {"state": "done", "result": {"predictions": answers, "unknown": []}}

    scored = evaluate(
        EvaluateRequest(run_id="r-gpu", dataset_dir=directory),
        Context(store=None, host=Host("http://gpu", "t"), client=Scores),
    )
    assert scored.where == "remote" and scored.samples == 2
    assert scored.metrics["exact_match"] == 1.0
