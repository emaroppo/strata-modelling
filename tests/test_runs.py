"""The run store: lineage, and the history query it exists for."""

import pytest

from strata.modelling import Run, RunStore


def a_run(**overrides) -> Run:
    base = dict(
        id=0,
        dataset="d",
        dataset_version=1,
        label_set="presence",
        model="counter.py:CountingModel",
        model_version="1",
        classes=["cat", "dog"],
    )
    return Run(**{**base, **overrides})


def test_a_recorded_run_reads_back(store):
    run = store.record(a_run(), {"accuracy": 0.5})
    assert store.get(run.id).classes == ["cat", "dog"]


def test_metrics_come_back_with_the_run(store):
    run = store.record(a_run(), {"accuracy": 0.5, "val_accuracy": 0.4})
    assert store.get(run.id).metrics == {"accuracy": 0.5, "val_accuracy": 0.4}


def test_an_unknown_run_is_none(store):
    assert store.get(999) is None


def test_latest_finds_the_newest_run_for_a_dataset(store):
    store.record(a_run(dataset_version=1), {})
    newest = store.record(a_run(dataset_version=2), {})
    assert store.latest("d").id == newest.id


def test_latest_is_per_dataset(store):
    mine = store.record(a_run(dataset="mine"), {})
    store.record(a_run(dataset="other"), {})
    assert store.latest("mine").id == mine.id


def test_latest_is_none_for_an_unknown_dataset(store):
    assert store.latest("never-trained") is None


def test_history_is_the_question_a_run_store_exists_for(store):
    for version, accuracy in enumerate([0.5, 0.7, 0.8], start=1):
        store.record(a_run(dataset_version=version), {"accuracy": accuracy})

    # Metric progression across rounds, as one query rather than a glob
    assert [value for _, _, value in store.history("d", "accuracy")] == [0.5, 0.7, 0.8]


def test_history_carries_the_dataset_version(store):
    store.record(a_run(dataset_version=4), {"accuracy": 0.9})
    assert store.history("d", "accuracy")[0][1] == 4


def test_history_of_an_unrecorded_metric_is_empty(store):
    store.record(a_run(), {"accuracy": 0.5})
    assert store.history("d", "f1") == []


def test_a_chain_of_one_is_just_the_run(store):
    run = store.record(a_run(), {})
    assert [r.id for r in store.chain(run.id)] == [run.id]


def test_a_chain_reads_oldest_first(store):
    first = store.record(a_run(), {})
    second = store.record(a_run(parent_run_id=first.id), {})
    assert [r.id for r in store.chain(second.id)] == [first.id, second.id]


def test_a_chain_survives_a_cycle(store):
    # Nothing should create one, but walking a chain must not hang if
    # something did
    first = store.record(a_run(), {})
    with store.engine.begin() as conn:
        from strata.modelling import tables as t

        conn.execute(t.run.update().where(t.run.c.id == first.id).values(parent_run_id=first.id))
    assert len(store.chain(first.id)) == 1


def test_checkpoint_paths_are_distinct_per_run(store):
    assert store.checkpoint_path(1) != store.checkpoint_path(2)


def test_deleting_a_run_takes_its_metrics(store):
    run = store.record(a_run(), {"accuracy": 0.5})
    store.delete(run.id)
    assert store.get(run.id) is None
    assert store.history("d", "accuracy") == []


def test_a_local_store_needs_no_infrastructure(tmp_path):
    store = RunStore.local(tmp_path / "runs")
    assert (tmp_path / "runs" / "runs.db").exists()
    assert store.checkpoints.is_dir()


def test_params_round_trip(store):
    run = store.record(a_run(params={"lr": 5e-5, "epochs": 4}), {})
    assert store.get(run.id).params == {"lr": 5e-5, "epochs": 4}


def test_metrics_are_floats_whatever_went_in(store):
    run = store.record(a_run(), {"n_train": 40})
    assert store.get(run.id).metrics["n_train"] == pytest.approx(40.0)
