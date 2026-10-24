"""A run keeps every checkpoint its model saved, and anything using one names it or gets the last.

``save_checkpoint`` is offered only to a ``finetune`` that takes it, so a
model that never asks is kept once, at the end, where it always was. A
checkpoint saved mid-training is put aside until the run has an id, then
moved under it. See ``docs/adr/0054``.
"""

import json
import re

import pytest
from counting_model import COUNTER

from strata.contracts import ChoicesPrediction
from strata.modelling import (
    PredictionCache,
    PredictRequest,
    TrainingError,
    TrainRequest,
    predict,
    train,
)
from strata.modelling.store.runs import FINAL, checkpoint_digest

SAVER = "saver.py:Saver"

#: A label model that keeps a checkpoint per epoch, and whose confidence is
#: the epoch it was saved at, so a prediction says which checkpoint made it.
SAVER_MODEL = """
import json
from pathlib import Path

from strata.contracts import ChoicesPrediction
from strata.modelling import LabelModel


class Saver(LabelModel):
    label_type = "classification"
    version = "1"

    def __init__(self, epochs: int = 3, fail_after: int = 0):
        self.epochs = epochs
        self.fail_after = fail_after
        self.classes = []
        self.epoch = 0

    def finetune(self, train, classes, val=None, on_epoch=None, save_checkpoint=None):
        self.classes = list(classes)
        started = self.epoch
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            if save_checkpoint is not None:
                save_checkpoint(f"epoch-{epoch}")
            if epoch == self.fail_after:
                raise RuntimeError("out of memory")
        return {"started_at": float(started)}

    def predict(self, paths, on_batch=None, *, features=None):
        return [
            ChoicesPrediction(values=self.classes[:1], confidences=[self.epoch / 10])
            for _ in paths
        ]

    def save(self, path):
        Path(path).write_text(json.dumps({"classes": self.classes, "epoch": self.epoch}))

    def load(self, path):
        payload = json.loads(Path(path).read_text())
        self.classes, self.epoch = payload["classes"], payload["epoch"]
"""


@pytest.fixture
def saver_dir(dataset_dir):
    def _make(**kwargs):
        root = dataset_dir(**kwargs)
        (root / SAVER.split(":")[0]).write_text(SAVER_MODEL)
        return root

    return _make


def _first_file(root):
    return sorted((root / "files").glob("*.jpg"))[0]


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------


def test_a_model_that_never_saves_is_kept_once_where_it_always_was(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)

    [only] = run.checkpoints
    assert only.name == FINAL
    assert only.path == store.checkpoint_path(run.id) == run.checkpoint
    assert only.digest == checkpoint_digest(only.path)


def test_save_checkpoint_is_offered_only_to_a_finetune_that_takes_it(store, saver_dir):
    root = saver_dir()
    # The counting model's finetune has no such parameter, and it was never
    # handed one: otherwise it would have failed on an unexpected keyword
    assert train(TrainRequest(dataset_dir=root, model=COUNTER), store).checkpoints[0].name == FINAL
    run = train(TrainRequest(dataset_dir=root, model=SAVER), store)
    assert [c.name for c in run.checkpoints] == ["epoch-1", "epoch-2", "epoch-3"]


def test_a_finetune_taking_any_keyword_is_offered_it(store, saver_dir):
    root = saver_dir()
    (root / "saver.py").write_text(
        (root / "saver.py")
        .read_text()
        .replace("on_epoch=None, save_checkpoint=None):", "on_epoch=None, **options):")
        .replace(
            "        started = self.epoch\n",
            "        started = self.epoch\n"
            '        save_checkpoint = options.get("save_checkpoint")\n',
        )
    )
    run = train(TrainRequest(dataset_dir=root, model=SAVER, params={"epochs": 2}), store)
    assert [c.name for c in run.checkpoints] == ["epoch-1", "epoch-2"]


def test_every_checkpoint_is_recorded_and_moved_under_the_run(store, saver_dir):
    run = train(TrainRequest(dataset_dir=saver_dir(), model=SAVER), store)

    for epoch, saved in enumerate(run.checkpoints, start=1):
        assert saved.path == store.checkpoint_path(run.id, f"epoch-{epoch}")
        assert json.loads(saved.path.read_text())["epoch"] == epoch
        assert saved.digest == checkpoint_digest(saved.path)
    # Nothing left where they were put aside while the run had no id
    assert not any((store.checkpoints / ".staging").iterdir())
    # As read back, oldest first, and the last is what a reader naming none gets
    assert store.get(run.id).checkpoints == run.checkpoints
    assert store.get(run.id).checkpoint == run.checkpoints[-1].path == run.checkpoint


def test_a_round_that_dies_leaves_no_run_and_no_checkpoints(store, saver_dir):
    with pytest.raises(RuntimeError, match="out of memory"):
        train(TrainRequest(dataset_dir=saver_dir(), model=SAVER, params={"fail_after": 2}), store)
    assert store.latest("d") is None
    assert not any((store.checkpoints / ".staging").iterdir())


def test_a_checkpoint_name_used_twice_is_refused(store, saver_dir):
    root = saver_dir()
    (root / "saver.py").write_text(
        (root / "saver.py").read_text().replace('f"epoch-{epoch}"', '"same"')
    )
    with pytest.raises(TrainingError, match="'same' was already saved"):
        train(TrainRequest(dataset_dir=root, model=SAVER), store)


def test_a_checkpoint_name_that_is_no_file_name_is_refused(store, saver_dir):
    root = saver_dir()
    (root / "saver.py").write_text(
        (root / "saver.py").read_text().replace('f"epoch-{epoch}"', 'f"../epoch {epoch}"')
    )
    with pytest.raises(TrainingError, match="becomes a file name"):
        train(TrainRequest(dataset_dir=root, model=SAVER), store)


def test_a_checkpoint_written_as_a_directory_is_digested_over_its_files(tmp_path):
    weights = tmp_path / "adapter"
    (weights / "nested").mkdir(parents=True)
    (weights / "adapter.bin").write_bytes(b"w")
    (weights / "nested" / "config.json").write_text("{}")
    first = checkpoint_digest(weights)

    (weights / "nested" / "config.json").write_text('{"r": 8}')

    assert first is not None and first != checkpoint_digest(weights)
    assert checkpoint_digest(tmp_path / "never-written") is None


# ----------------------------------------------------------------------
# Using one
# ----------------------------------------------------------------------


def test_predict_uses_the_last_checkpoint_unless_one_is_named(store, saver_dir):
    root = saver_dir()
    run = train(TrainRequest(dataset_dir=root, model=SAVER), store)

    [last] = predict(PredictRequest(run_id=run.id, paths=[_first_file(root)]), store)
    [first] = predict(
        PredictRequest(run_id=run.id, checkpoint="epoch-1", paths=[_first_file(root)]), store
    )

    assert last.value.confidences == [pytest.approx(0.3)]
    assert first.value.confidences == [pytest.approx(0.1)]


def test_predicting_from_a_checkpoint_the_run_never_kept_is_refused(store, saver_dir):
    run = train(TrainRequest(dataset_dir=saver_dir(), model=SAVER), store)
    with pytest.raises(TrainingError, match=re.escape("no checkpoint named 'epoch-7'")):
        predict(PredictRequest(run_id=run.id, checkpoint="epoch-7", paths=[]), store)


def test_a_warm_start_continues_from_the_checkpoint_it_names(store, saver_dir):
    first = train(TrainRequest(dataset_dir=saver_dir(), model=SAVER), store)
    request = TrainRequest(
        dataset_dir=saver_dir(version=2), model=SAVER, parent_run_id=first.id, params={"epochs": 1}
    )

    named = train(request.model_copy(update={"parent_checkpoint": "epoch-2"}), store)
    last = train(request, store)

    assert named.metrics["started_at"] == 2
    assert last.metrics["started_at"] == 3


def test_a_warm_start_from_a_checkpoint_the_parent_never_kept_is_refused(store, saver_dir):
    first = train(TrainRequest(dataset_dir=saver_dir(), model=SAVER), store)
    with pytest.raises(TrainingError, match="no checkpoint named 'best'"):
        train(
            TrainRequest(
                dataset_dir=saver_dir(version=2),
                model=SAVER,
                parent_run_id=first.id,
                parent_checkpoint="best",
            ),
            store,
        )


# ----------------------------------------------------------------------
# The prediction cache
# ----------------------------------------------------------------------


def _guess(confidence: float) -> ChoicesPrediction:
    return ChoicesPrediction(values=["cat"], confidences=[confidence])


def test_two_checkpoints_do_not_share_a_cached_prediction(store, saver_dir):
    run = train(TrainRequest(dataset_dir=saver_dir(), model=SAVER), store)
    cache = PredictionCache.beside(store)

    cache.put(run.id, {"a" * 64: _guess(0.1)}, checkpoint="epoch-1")

    # One run, one checksum, one digest: only the weights differ, and a
    # prediction is a function of the weights. docs/adr/0006
    assert cache.get(run.id, ["a" * 64], checkpoint="epoch-2") == {}
    assert cache.get(run.id, ["a" * 64], checkpoint="epoch-1")["a" * 64].confidences == [0.1]

    cache.put(run.id, {"a" * 64: _guess(0.2)}, checkpoint="epoch-2")
    assert cache.counts() == {run.id: 2}


def test_the_cache_reads_and_writes_the_last_checkpoint_when_none_is_named(store, saver_dir):
    run = train(TrainRequest(dataset_dir=saver_dir(), model=SAVER), store)
    cache = PredictionCache.beside(store)

    cache.put(run.id, {"a" * 64: _guess(0.3)})

    assert cache.last_checkpoint(run.id) == "epoch-3"
    assert cache.get(run.id, ["a" * 64], checkpoint="epoch-3")["a" * 64].confidences == [0.3]
    assert cache.get(run.id, ["a" * 64], checkpoint="epoch-1") == {}


# ----------------------------------------------------------------------
# Merging
# ----------------------------------------------------------------------


def test_a_merge_brings_every_checkpoint_when_asked(tmp_path, store, saver_dir):
    from strata.modelling import RunStore, merge_stores

    run = train(TrainRequest(dataset_dir=saver_dir(), model=SAVER), store)
    target = RunStore.local(tmp_path / "elsewhere")

    report = merge_stores(store, target, checkpoints=True)

    merged = target.get(run.id)
    assert report.checkpoints == 3
    assert [c.name for c in merged.checkpoints] == ["epoch-1", "epoch-2", "epoch-3"]
    assert [c.digest for c in merged.checkpoints] == [c.digest for c in run.checkpoints]
    assert merged.checkpoint == target.checkpoint_path(run.id, "epoch-3")
    assert json.loads(merged.checkpoint.read_text())["epoch"] == 3


def test_a_merge_without_checkpoints_keeps_their_record_and_claims_no_file(
    tmp_path, store, saver_dir
):
    from strata.modelling import RunStore, merge_stores

    run = train(TrainRequest(dataset_dir=saver_dir(), model=SAVER), store)
    target = RunStore.local(tmp_path / "elsewhere")

    merge_stores(store, target)

    merged = target.get(run.id)
    assert [(c.name, c.path) for c in merged.checkpoints] == [
        ("epoch-1", None),
        ("epoch-2", None),
        ("epoch-3", None),
    ]
    assert merged.checkpoint is None
