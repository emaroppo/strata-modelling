"""The run store: lineage, and the history query it exists for."""

import pytest
from run_factory import a_run

from strata.modelling import RunStore


def test_a_recorded_run_reads_back(store):
    run = store.record(a_run(classes=["cat", "dog"]), {"accuracy": 0.5})
    assert store.get(run.id).classes == ["cat", "dog"]


def test_metrics_come_back_with_the_run(store):
    run = store.record(a_run(), {"accuracy": 0.5, "val_accuracy": 0.4})
    assert store.get(run.id).metrics == {"accuracy": 0.5, "val_accuracy": 0.4}


def test_an_unknown_run_is_none(store):
    assert store.get(999) is None


def test_latest_finds_the_newest_run_for_a_dataset(store):
    store.record(a_run(dataset_version=1), {})
    newest = store.record(a_run(dataset_version=2), {})
    assert store.latest("demo").id == newest.id


def test_latest_is_per_dataset(store):
    mine = store.record(a_run(dataset="mine"), {})
    store.record(a_run(dataset="other"), {})
    assert store.latest("mine").id == mine.id


def test_latest_is_none_for_an_unknown_dataset(store):
    assert store.latest("never-trained") is None


def test_latest_does_not_reach_back_past_a_re_split(store):
    # Versions 1 and 2 share sides; version 3 re-split from nothing, so a
    # run over 1 or 2 may have trained on what 3 holds out
    store.record(a_run(dataset_version=1), {})
    over_two = store.record(a_run(dataset_version=2), {})
    unversioned = store.record(a_run(dataset_version=None), {})
    assert store.latest("demo").id == unversioned.id
    assert store.latest("demo", since_version=1).id == over_two.id
    # Nothing since the re-split yet, and a run that cannot say which
    # version it saw is not continued either
    assert store.latest("demo", since_version=3) is None
    over_three = store.record(a_run(dataset_version=3), {})
    assert store.latest("demo", since_version=3).id == over_three.id


def test_history_is_the_question_a_run_store_exists_for(store):
    for version, accuracy in enumerate([0.5, 0.7, 0.8], start=1):
        store.record(a_run(dataset_version=version), {"accuracy": accuracy})

    # Metric progression across rounds, as one query rather than a glob
    assert [value for _, _, value in store.history("demo", "accuracy")] == [0.5, 0.7, 0.8]


def test_history_carries_the_dataset_version(store):
    store.record(a_run(dataset_version=4), {"accuracy": 0.9})
    assert store.history("demo", "accuracy")[0][1] == 4


def test_history_of_an_unrecorded_metric_is_empty(store):
    store.record(a_run(), {"accuracy": 0.5})
    assert store.history("demo", "f1") == []


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
        from strata.modelling.store import tables as t

        conn.execute(t.run.update().where(t.run.c.id == first.id).values(parent_run_id=first.id))
    assert len(store.chain(first.id)) == 1


def test_checkpoint_paths_are_distinct_per_run(store):
    assert store.checkpoint_path(1) != store.checkpoint_path(2)


def test_deleting_a_run_takes_its_metrics(store):
    run = store.record(a_run(), {"accuracy": 0.5})
    store.delete(run.id)
    assert store.get(run.id) is None
    assert store.history("demo", "accuracy") == []


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


# ----------------------------------------------------------------------
# the training curve
# ----------------------------------------------------------------------


def test_a_run_records_what_it_reported_as_it_trained(tmp_path):
    store = RunStore.local(tmp_path)
    run = store.record(
        a_run(),
        {"val_accuracy": 0.9},
        curve=[(1, {"loss": 0.5}), (2, {"loss": 0.3})],
    )

    assert store.curve(run.id) == [(1, {"loss": 0.5}), (2, {"loss": 0.3})]


def test_the_curve_is_not_mixed_into_the_final_metrics(tmp_path):
    """`get` answers with what the run ended at, not every step of it."""
    store = RunStore.local(tmp_path)
    run = store.record(
        a_run(),
        {"loss": 0.2},
        curve=[(1, {"loss": 0.9}), (2, {"loss": 0.4})],
    )

    assert store.get(run.id).metrics == {"loss": 0.2}


def test_a_run_that_reported_nothing_has_no_curve(tmp_path):
    """A model may ignore on_epoch, and silence is not a zero-length epoch."""
    store = RunStore.local(tmp_path)
    run = store.record(
        a_run(),
        {"loss": 0.2},
    )

    assert store.curve(run.id) == []


def test_a_curve_does_not_reach_the_history(tmp_path):
    """`history` plots one point per run, not one per epoch."""
    store = RunStore.local(tmp_path)
    store.record(
        a_run(),
        {"val_accuracy": 0.9},
        curve=[(1, {"val_accuracy": 0.1}), (2, {"val_accuracy": 0.5})],
    )

    assert [v for _, _, v in store.history("demo", "val_accuracy")] == [0.9]


def test_a_run_says_how_much_of_what_it_saw_nobody_checked(store):
    from strata.modelling import Seen, Unchecked

    run = store.record(
        a_run(),
        {},
        saw=[
            Seen("a" * 64, "train", "batch-1", False),
            Seen("b" * 64, "train", "batch-1", True),
            Seen("c" * 64, "train", None, True),
            Seen("d" * 64, "val", "batch-1", False),
            Seen("e" * 64, "val", "batch-2", False),
            # The manifest said nothing: unknown, which is not unreviewed
            Seen("f" * 64, "val", "batch-2", None),
        ],
    )
    assert store.unchecked(run.id) == [
        Unchecked("train", None, 1, 0),
        Unchecked("train", "batch-1", 2, 1),
        Unchecked("val", "batch-1", 1, 1),
        Unchecked("val", "batch-2", 2, 1),
    ]


def test_a_run_from_before_this_was_written_down_says_nothing(store):
    run = store.record(a_run(), {}, saw=[("a" * 64, "train")])
    from strata.modelling import Unchecked

    # A side without batches or reviews still counts, as unknown rather than unreviewed
    assert store.unchecked(run.id) == [Unchecked("train", None, 1, 0)]
    bare = store.record(a_run(), {})
    assert store.unchecked(bare.id) == []


def test_opening_a_store_that_does_not_exist_creates_nothing(tmp_path):
    from strata.modelling import RunStoreMissing

    root = tmp_path / "runs"
    with pytest.raises(RunStoreMissing, match="No runs recorded"):
        RunStore.open(root)
    assert not root.exists()


def test_local_makes_a_store_and_open_reads_it_back(tmp_path):
    root = tmp_path / "runs"
    made = RunStore.local(root)
    run = made.record(a_run(), {"accuracy": 0.5})
    assert RunStore.open(root).get(run.id) == run


def test_a_store_behind_the_code_is_refused_on_open(tmp_path):
    from sqlalchemy import create_engine

    from strata.common.migrations import SchemaOutOfDate

    root = tmp_path / "runs"
    RunStore.local(root)
    with create_engine(f"sqlite:///{root / 'runs.db'}").begin() as conn:
        conn.exec_driver_sql("DELETE FROM alembic_version")
    with pytest.raises(SchemaOutOfDate, match="upgrade head"):
        RunStore.open(root)
