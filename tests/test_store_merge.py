"""Putting a split history back together.

A project's rounds are spread across two stores: the early ones ran on the
desktop, the later ones on the GPU host. Reading the curve means having
both in one place.

What makes this possible at all is that run ids are timestamps and host
tokens rather than integers. Under the old numbering, both stores called
their first run 1, and there was no merge to write — only a collision.
"""

import pytest
from run_factory import recorded

from strata.modelling import RunStore, StoreMergeError, merge_stores


@pytest.fixture
def stores(tmp_path):
    return RunStore.local(tmp_path / "gpu"), RunStore.local(tmp_path / "desktop")


# ----------------------------------------------------------------------
# The runs
# ----------------------------------------------------------------------


def test_runs_come_across(stores):
    source, target = stores
    theirs = recorded(source, origin="gpu-host")
    recorded(target, origin="desktop")

    report = merge_stores(source, target)

    assert report.runs == 1
    assert target.get(theirs.id).origin == "gpu-host"
    assert len(target.history("demo", "val_accuracy")) == 2


def test_what_a_run_saw_comes_with_it(stores):
    from strata.modelling import Run

    source, target = stores
    run = source.record(
        Run(id="", dataset="demo", label_set="l", model="m", model_version="1"),
        {"val_accuracy": 0.5},
        saw=[("aaa", "train"), ("bbb", "val"), ("ccc", "holdout")],
    )

    report = merge_stores(source, target)

    assert report.samples == 3
    assert target.saw(run.id) == {"train": ["aaa"], "val": ["bbb"], "holdout": ["ccc"]}
    assert "3 sample side(s)" in report.lines()


def test_metrics_come_with_them(stores):
    source, target = stores
    run = recorded(source, metrics={"val_accuracy": 0.9, "loss": 0.1})

    merge_stores(source, target)

    # A run without its numbers is a row nobody can read anything from
    assert report_metrics(target, run.id) == {"val_accuracy": 0.9, "loss": 0.1}


def report_metrics(store, run_id):
    from sqlalchemy import select

    from strata.modelling.store import tables as t

    with store.engine.connect() as conn:
        return {
            row.name: row.value
            for row in conn.execute(select(t.metric).where(t.metric.c.run_id == run_id))
            if row.epoch is None
        }


def test_merging_twice_copies_nothing_the_second_time(stores):
    source, target = stores
    recorded(source)

    merge_stores(source, target)
    second = merge_stores(source, target)

    # Ids are globally unique, so an id already there is the same run —
    # which is what makes an interrupted merge safe to repeat
    assert (second.runs, second.already_present) == (0, 1)
    assert len(target.history("demo", "val_accuracy")) == 1


def test_the_same_store_twice_is_refused(stores):
    source, _ = stores
    with pytest.raises(StoreMergeError, match="same store"):
        merge_stores(source, source)


# ----------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------


def test_a_chain_survives_the_move(stores):
    source, target = stores
    first = recorded(source)
    second = recorded(source, parent_run_id=first.id)

    merge_stores(source, target)

    # Rounds warm-start from each other, so the chain is the history. A
    # merge that dropped it would turn every run into an unrelated cold one.
    assert [r.id for r in target.chain(second.id)] == [first.id, second.id]


def test_a_parent_already_in_the_target_is_honoured(stores):
    source, target = stores
    # The desktop trained the first round; the GPU host continued from it,
    # which is exactly how the split happened
    first = recorded(target)
    second = recorded(source, parent_run_id=first.id)

    report = merge_stores(source, target)

    assert report.orphaned == []
    assert target.get(second.id).parent_run_id == first.id


def test_a_parent_in_neither_store_is_reported_not_dangled(stores):
    source, target = stores
    orphan = recorded(source, parent_run_id="20250101T000000-deadbeef")

    report = merge_stores(source, target)

    # Kept, because the run happened; the link dropped, because a lineage
    # that cannot be shown is a claim the history cannot support
    assert report.orphaned == [orphan.id]
    assert target.get(orphan.id) is not None
    assert target.get(orphan.id).parent_run_id is None


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def test_a_checkpoint_is_left_behind_by_default(stores, tmp_path):
    source, target = stores
    run = recorded(source)
    checkpoint = source.checkpoint_path(run.id)
    checkpoint.write_bytes(b"weights")
    _attach(source, run.id, checkpoint)

    report = merge_stores(source, target)

    # Not copied, and — the part that matters — not claimed either. Every
    # caller that warm-starts or predicts reads this column, and a path to
    # a file on another machine fails at the far end of a long round.
    assert report.checkpoints == 0
    assert target.get(run.id).checkpoint is None


def test_a_checkpoint_comes_when_asked_for(stores):
    source, target = stores
    run = recorded(source)
    checkpoint = source.checkpoint_path(run.id)
    checkpoint.write_bytes(b"weights")
    _attach(source, run.id, checkpoint)

    report = merge_stores(source, target, checkpoints=True)

    moved = target.get(run.id).checkpoint
    assert report.checkpoints == 1
    # Rewritten to the target's own layout, not the source's path
    assert moved == target.checkpoint_path(run.id)
    assert target.checkpoint_path(run.id).read_bytes() == b"weights"


def test_a_checkpoint_that_is_gone_is_not_claimed(stores):
    source, target = stores
    run = recorded(source)
    _attach(source, run.id, source.checkpoint_path(run.id))  # never written

    report = merge_stores(source, target, checkpoints=True)

    assert report.checkpoints == 0
    assert target.get(run.id).checkpoint is None


def _attach(store, run_id, path):
    from sqlalchemy import update

    from strata.modelling.store import tables as t

    with store.engine.begin() as conn:
        conn.execute(update(t.run).where(t.run.c.id == run_id).values(checkpoint=str(path)))


# ----------------------------------------------------------------------
# Predictions, and the dry run
# ----------------------------------------------------------------------


def test_cached_predictions_come_across(stores):
    from strata.labels import ChoicesPrediction
    from strata.modelling import PredictionCache

    source, target = stores
    run = recorded(source)
    PredictionCache.beside(source).put(
        run.id, {"abc123": ChoicesPrediction(values=["cat"], confidences=[0.9])}
    )

    report = merge_stores(source, target)

    # Scoring a pool is minutes of GPU and nothing invalidates a prediction,
    # so leaving the cache behind would mean paying for it twice
    assert report.predictions == 1
    assert PredictionCache.beside(target).get(run.id, ["abc123"])


def test_a_dry_run_writes_nothing(stores):
    source, target = stores
    run = recorded(source)

    report = merge_stores(source, target, dry_run=True)

    assert report.runs == 1
    assert target.get(run.id) is None
