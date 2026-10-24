"""The evaluate stage scores every checkpoint a run kept, and a media model's by what it makes.

A label model's checkpoints are each predicted with, the cache keeping
them apart. A media model's are each generated from, one output per
sample's input with that sample's seed, and the outputs are scored beside
the side's targets and the training side's, then kept in the result. A
request that uses none of this is keyed as it was before it existed. See
``docs/adr/0054`` and ``docs/adr/0037``.
"""

import json
from pathlib import Path
from typing import ClassVar

import pytest
from counting_model import COUNTER
from media_model import ECHO
from test_modelling_stages import FakeHost

from strata.common.canonical import content_hash
from strata.contracts import MANIFEST_NAME
from strata.modelling import GeneratedOutput, PredictionCache
from strata.modelling.stages import (
    Context,
    EvaluateRequest,
    Host,
    StageError,
    TaskRef,
    TrainStageRequest,
    evaluate,
    sample_seed,
    train,
)

LEARNER = "learner.py:Learner"

#: A label model that keeps a checkpoint per epoch and gets better: it says
#: "dog" after its first epoch and "cat" after every later one, so each
#: checkpoint scores differently on a side of cats.
LEARNER_MODEL = """
import json
from pathlib import Path

from strata.contracts import ChoicesPrediction
from strata.modelling import LabelModel


class Learner(LabelModel):
    label_type = "classification"
    version = "1"

    def __init__(self, epochs: int = 3):
        self.epochs = epochs
        self.epoch = 0
        self.started_at = 0

    def finetune(self, train, classes, val=None, on_epoch=None, save_checkpoint=None):
        started = self.epoch
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            if save_checkpoint is not None:
                save_checkpoint(f"epoch-{epoch}")
        return {"started_at": float(started)}

    def predict(self, paths, on_batch=None, *, features=None):
        guess = "dog" if self.epoch == 1 else "cat"
        return [ChoicesPrediction(values=[guess], confidences=[0.9]) for _ in paths]

    def save(self, path):
        Path(path).write_text(json.dumps({"epoch": self.epoch}))

    def load(self, path):
        self.epoch = json.loads(Path(path).read_text())["epoch"]
"""


def _learner_run(store, dataset_dir, held: int = 3):
    directory = dataset_dir(n_train=8, n_val=2)
    (directory / LEARNER.split(":")[0]).write_text(LEARNER_MODEL)
    _hold_out(directory, held)
    record = train(
        TrainStageRequest(dataset_dir=directory, model=LEARNER, fresh=True), Context(store)
    )
    return directory, record.run_id


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


def _echo_run(store, sft_dir, **made):
    directory = sft_dir(**made)
    record = train(TrainStageRequest(dataset_dir=directory, model=ECHO, fresh=True), Context(store))
    return directory, record.run_id


def _val(directory) -> list[dict]:
    payload = json.loads((directory / MANIFEST_NAME).read_text())
    return [s for s in payload["samples"] if s["split"] == "val"]


# ----------------------------------------------------------------------
# A label model: every checkpoint predicted with
# ----------------------------------------------------------------------


def test_every_checkpoint_of_a_label_models_run_is_scored_and_named(store, dataset_dir):
    directory, run_id = _learner_run(store, dataset_dir)

    scored = evaluate(EvaluateRequest(run_id=run_id, dataset_dir=directory), Context(store))

    kept = store.get(run_id).checkpoints
    assert [(c.name, c.digest) for c in scored.checkpoints] == [(c.name, c.digest) for c in kept]
    assert [c.scores["classify"].metrics["exact_match"] for c in scored.checkpoints] == [
        0.0,
        1.0,
        1.0,
    ]
    # What a reader of one number reads is the run's last, and says so
    assert scored.checkpoint == "epoch-3"
    assert scored.scores == scored.checkpoints[-1].scores
    assert scored.metrics["exact_match"] == 1.0
    # A label model makes nothing to keep
    assert all(not c.outputs and c.outputs_file is None for c in scored.checkpoints)


def test_each_checkpoints_predictions_are_cached_apart(store, dataset_dir, monkeypatch):
    directory, run_id = _learner_run(store, dataset_dir, held=3)
    request = EvaluateRequest(run_id=run_id, dataset_dir=directory)
    first = evaluate(request, Context(store))
    assert PredictionCache.beside(store).counts() == {run_id: 3 * 3}

    import strata.modelling.handlers as handlers

    monkeypatch.setattr(handlers, "predict", lambda *a, **k: pytest.fail("predicted again"))
    assert evaluate(request, Context(store)).checkpoints == first.checkpoints


def test_a_request_can_score_only_the_checkpoints_it_names(store, dataset_dir):
    directory, run_id = _learner_run(store, dataset_dir)
    scored = evaluate(
        EvaluateRequest(run_id=run_id, dataset_dir=directory, checkpoints=["epoch-1"]),
        Context(store),
    )
    assert [c.name for c in scored.checkpoints] == ["epoch-1"]
    assert scored.checkpoint == "epoch-1" and scored.metrics["exact_match"] == 0.0


def test_named_checkpoints_are_scored_oldest_first_whatever_order_they_are_named_in(
    store, dataset_dir
):
    directory, run_id = _learner_run(store, dataset_dir)
    scored = evaluate(
        EvaluateRequest(run_id=run_id, dataset_dir=directory, checkpoints=["epoch-3", "epoch-1"]),
        Context(store),
    )
    assert [c.name for c in scored.checkpoints] == ["epoch-1", "epoch-3"]


def test_a_checkpoint_the_run_never_kept_is_refused_by_name(store, dataset_dir, monkeypatch):
    directory, run_id = _learner_run(store, dataset_dir)
    import strata.modelling.handlers as handlers

    monkeypatch.setattr(handlers, "predict", lambda *a, **k: pytest.fail("predicted"))
    request = EvaluateRequest(run_id=run_id, dataset_dir=directory, checkpoints=["epoch-9"])
    with pytest.raises(
        StageError, match=r"no checkpoint named 'epoch-9'\. It kept: epoch-1, epoch-2"
    ):
        evaluate(request, Context(store))


def test_a_run_the_store_does_not_hold_is_refused(store, dataset_dir):
    directory = dataset_dir()
    with pytest.raises(StageError, match="No run 'nowhere'"):
        evaluate(EvaluateRequest(run_id="nowhere", dataset_dir=directory), Context(store))


def test_a_label_models_checkpoints_are_each_predicted_by_the_host(dataset_dir):
    directory = dataset_dir(n_train=8, n_val=2)
    _hold_out(directory, 2)
    held = [s["checksum"] for s in json.loads((directory / MANIFEST_NAME).read_text())["samples"]]
    asked = []

    class Two(FakeHost):
        held: ClassVar[dict] = {
            **FakeHost.held,
            "checkpoints": [{"name": "epoch-1"}, {"name": "epoch-2"}],
        }

        def predict(self, request):
            asked.append(request.checkpoint)
            return {"id": "job-2"}

        def follow(self, job_id, on_state=None):
            cat = {"kind": "choices", "values": ["cat"], "confidences": [0.9]}
            return {"state": "done", "result": {"predictions": dict.fromkeys(held, cat)}}

    scored = evaluate(
        EvaluateRequest(run_id="r-gpu", dataset_dir=directory),
        Context(store=None, host=Host("http://gpu", "t"), client=Two),
    )
    assert asked == ["epoch-1", "epoch-2"]
    assert [c.name for c in scored.checkpoints] == ["epoch-1", "epoch-2"]
    assert scored.where == "remote"


def test_a_host_older_than_checkpoints_is_scored_once_from_its_last(dataset_dir):
    directory = dataset_dir(n_train=8, n_val=2)
    _hold_out(directory, 2)
    held = [s["checksum"] for s in json.loads((directory / MANIFEST_NAME).read_text())["samples"]]

    class Old(FakeHost):
        held: ClassVar[dict] = {k: v for k, v in FakeHost.held.items() if k != "checkpoints"}

        def predict(self, request):
            assert request.checkpoint is None
            return {"id": "job-2"}

        def follow(self, job_id, on_state=None):
            cat = {"kind": "choices", "values": ["cat"], "confidences": [0.9]}
            return {"state": "done", "result": {"predictions": dict.fromkeys(held, cat)}}

    scored = evaluate(
        EvaluateRequest(run_id="r-gpu", dataset_dir=directory),
        Context(store=None, host=Host("http://gpu", "t"), client=Old),
    )
    assert [c.name for c in scored.checkpoints] == [None] and scored.checkpoint is None
    assert scored.metrics["exact_match"] == 1.0


def test_a_host_run_whose_model_does_not_resolve_here_is_scored_as_a_label_model(dataset_dir):
    """The host predicts, so its model need not import here: a base install, say."""
    directory = dataset_dir(n_train=8, n_val=2)
    _hold_out(directory, 2)
    held = [s["checksum"] for s in json.loads((directory / MANIFEST_NAME).read_text())["samples"]]

    class Elsewhere(FakeHost):
        held: ClassVar[dict] = {**FakeHost.held, "model": "only-on-the-host"}

        def predict(self, request):
            return {"id": "job-2"}

        def follow(self, job_id, on_state=None):
            cat = {"kind": "choices", "values": ["cat"], "confidences": [0.9]}
            return {"state": "done", "result": {"predictions": dict.fromkeys(held, cat)}}

    scored = evaluate(
        EvaluateRequest(run_id="r-gpu", dataset_dir=directory),
        Context(store=None, host=Host("http://gpu", "t"), client=Elsewhere),
    )
    assert scored.metrics["exact_match"] == 1.0


# ----------------------------------------------------------------------
# A media model: every checkpoint generated from
# ----------------------------------------------------------------------


def test_a_media_models_run_is_scored_at_every_checkpoint_by_what_it_generates(store, sft_dir):
    directory, run_id = _echo_run(store, sft_dir)

    scored = evaluate(
        EvaluateRequest(run_id=run_id, dataset_dir=directory, side="val"), Context(store)
    )

    val = _val(directory)
    assert scored.samples == len(val) == 2
    assert [c.name for c in scored.checkpoints] == ["epoch-1", "epoch-2", "epoch-3"]
    for epoch, checkpoint in enumerate(scored.checkpoints, start=1):
        # Scored by its media's task when none is named
        assert set(checkpoint.scores) == {"text-generation"}
        score = checkpoint.scores["text-generation"]
        assert score.counts["outputs"] == 2 and len(score.detail["longest_run"]) == 2
        echoed = [json.loads(o.output) for o in checkpoint.outputs]
        # Each made from its own sample's input, by the checkpoint named
        assert [e["values"] for e in echoed] == [s["features"] for s in val]
        assert {e["epoch"] for e in echoed} == {epoch}
        assert [o.checksum for o in checkpoint.outputs] == [s["checksum"] for s in val]
    assert scored.checkpoint == "epoch-3" and scored.scores == scored.checkpoints[-1].scores
    # Classification numbers belong to a label model
    assert scored.metrics == {} and scored.per_class == {}


def test_each_sample_is_generated_with_its_own_seed_at_every_checkpoint(store, sft_dir):
    directory, run_id = _echo_run(store, sft_dir)
    request = EvaluateRequest(run_id=run_id, dataset_dir=directory, side="val", seed=7)

    scored = evaluate(request, Context(store))

    wanted = [sample_seed(7, s["checksum"]) for s in _val(directory)]
    for checkpoint in scored.checkpoints:
        assert [o.seed for o in checkpoint.outputs] == wanted
        assert [json.loads(o.output)["seed"] for o in checkpoint.outputs] == wanted
        # The echo model ends on an even seed, and the task counts it
        ended = [seed % 2 == 0 for seed in wanted]
        assert [o.ended for o in checkpoint.outputs] == ended
        assert checkpoint.scores["text-generation"].metrics["ended"] == sum(ended) / 2
    # Another stage seed is other seeds
    assert wanted != [sample_seed(0, s["checksum"]) for s in _val(directory)]


def test_a_sample_keeps_its_seed_whatever_else_is_on_the_side():
    assert sample_seed(3, "a" * 64) == sample_seed(3, "a" * 64)
    assert sample_seed(3, "a" * 64) != sample_seed(3, "b" * 64)
    assert 0 <= sample_seed(3, "a" * 64) < 2**31


def test_generation_uses_the_models_own_settings_unless_the_stage_gives_some(store, sft_dir):
    directory, run_id = _echo_run(store, sft_dir)
    plain = evaluate(
        EvaluateRequest(run_id=run_id, dataset_dir=directory, side="val"), Context(store)
    )
    # Nothing given: the model fills in its own
    assert all(json.loads(o.output)["settings"] == {} for o in plain.checkpoints[0].outputs)

    given = evaluate(
        EvaluateRequest(
            run_id=run_id, dataset_dir=directory, side="val", settings={"temperature": 0.5}
        ),
        Context(store),
    )
    for checkpoint in given.checkpoints:
        assert all(
            json.loads(o.output)["settings"] == {"temperature": 0.5} for o in checkpoint.outputs
        )


def test_memorisation_is_measured_against_the_training_side_not_the_side_scored(store, sft_dir):
    directory, run_id = _echo_run(store, sft_dir)
    # A training body that the first output repeats four words of, and a
    # validation body it repeats more of, which must not count
    (directory / "files" / "body000.txt").write_text("the quick brown fox")
    path = directory / MANIFEST_NAME
    payload = json.loads(path.read_text())
    val = [s for s in payload["samples"] if s["split"] == "val"]
    val[0]["features"] = {"subject": "x the quick brown fox jumps over y"}
    (directory / val[1]["path"]).write_text("the quick brown fox jumps over")
    path.write_text(json.dumps(payload))

    scored = evaluate(
        EvaluateRequest(run_id=run_id, dataset_dir=directory, side="val"), Context(store)
    )
    score = scored.checkpoints[0].scores["text-generation"]
    assert score.detail["longest_run"] == [4, 0]
    assert score.metrics["longest_run_max"] == 4.0


def test_agreement_with_the_target_is_scored_when_asked(store, sft_dir):
    directory, run_id = _echo_run(store, sft_dir)
    scored = evaluate(
        EvaluateRequest(
            run_id=run_id,
            dataset_dir=directory,
            side="val",
            tasks=[TaskRef(ref="text-generation", params={"chrf": True})],
        ),
        Context(store),
    )
    assert all("chrf" in c.scores["text-generation"].metrics for c in scored.checkpoints)


def test_outputs_stay_in_the_record_where_a_person_can_read_them(store, sft_dir):
    directory, run_id = _echo_run(store, sft_dir)
    scored = evaluate(
        EvaluateRequest(run_id=run_id, dataset_dir=directory, side="val"), Context(store)
    )
    dumped = scored.model_dump(mode="json")
    [first, *_] = dumped["checkpoints"][0]["outputs"]
    assert json.loads(first["output"])["values"] == {"subject": "subject 4"}
    assert dumped["checkpoints"][0]["outputs_file"] is None


def test_outputs_are_kept_beside_the_record_in_a_file_named_by_their_content(
    store, sft_dir, tmp_path
):
    directory, run_id = _echo_run(store, sft_dir)
    request = EvaluateRequest(run_id=run_id, dataset_dir=directory, side="val")
    beside = tmp_path / "outputs"

    scored = evaluate(request, Context(store, outputs=beside))

    for checkpoint in scored.checkpoints:
        assert checkpoint.outputs == [] and checkpoint.outputs_file is not None
        lines = (beside / checkpoint.outputs_file).read_text().splitlines()
        assert checkpoint.outputs_file == f"{content_hash([json.loads(x) for x in lines])}.jsonl"
        assert [json.loads(json.loads(x)["output"])["epoch"] for x in lines] == [
            int(checkpoint.name.removeprefix("epoch-"))
        ] * 2
    assert len({c.outputs_file for c in scored.checkpoints}) == 3
    # Made again, the same record names the same files
    assert evaluate(request, Context(store, outputs=beside)) == scored
    assert len(list(beside.iterdir())) == 3


def test_only_the_checkpoints_named_are_generated_from(store, sft_dir):
    directory, run_id = _echo_run(store, sft_dir)
    scored = evaluate(
        EvaluateRequest(run_id=run_id, dataset_dir=directory, side="val", checkpoints=["epoch-2"]),
        Context(store),
    )
    [only] = scored.checkpoints
    assert only.name == "epoch-2"
    assert {json.loads(o.output)["epoch"] for o in only.outputs} == {2}


def test_a_media_models_checkpoints_are_each_generated_from_on_the_host(store, sft_dir):
    directory, run_id = _echo_run(store, sft_dir)
    model = store.get(run_id).model
    asked = []

    class Generating(FakeHost):
        held: ClassVar[dict] = {
            **FakeHost.held,
            "model": model,
            "classes": [],
            "checkpoints": [{"name": "epoch-1"}, {"name": "epoch-2"}],
        }

        def generate(self, request):
            asked.append(request)
            return {"id": "job-4"}

        def follow(self, job_id, on_state=None):
            request = asked[-1]
            outputs = [
                GeneratedOutput(
                    data=f"{request.checkpoint}: {i.values['subject']}".encode(),
                    ended=True,
                    settings_digest=content_hash(s),
                    seed=seed,
                    checkpoint=request.checkpoint,
                ).model_dump(mode="json")
                for i, s, seed in zip(request.inputs, request.settings, request.seeds, strict=True)
            ]
            return {"state": "done", "result": {"outputs": outputs}}

    scored = evaluate(
        EvaluateRequest(run_id=run_id, dataset_dir=directory, side="val"),
        Context(store=None, host=Host("http://gpu", "t"), client=Generating),
    )
    val = _val(directory)
    assert [r.checkpoint for r in asked] == ["epoch-1", "epoch-2"]
    assert [i.values for i in asked[0].inputs] == [s["features"] for s in val]
    assert asked[0].seeds == [sample_seed(0, s["checksum"]) for s in val]
    assert [o.output for o in scored.checkpoints[1].outputs] == [
        "epoch-2: subject 4",
        "epoch-2: subject 5",
    ]
    assert scored.where == "remote"


# ----------------------------------------------------------------------
# Which tasks a kind of model is scored by
# ----------------------------------------------------------------------


def test_a_generation_task_for_a_label_models_run_is_refused(store, dataset_dir, monkeypatch):
    directory, run_id = _learner_run(store, dataset_dir)
    import strata.modelling.handlers as handlers

    monkeypatch.setattr(handlers, "predict", lambda *a, **k: pytest.fail("predicted"))
    request = EvaluateRequest(
        run_id=run_id, dataset_dir=directory, tasks=[TaskRef(ref="text-generation")]
    )
    with pytest.raises(StageError, match=r"scores what a media model generates.*predicts labels"):
        evaluate(request, Context(store))


def test_a_label_task_for_a_media_models_run_is_refused(store, sft_dir, monkeypatch):
    directory, run_id = _echo_run(store, sft_dir)
    import strata.modelling.handlers as handlers

    monkeypatch.setattr(handlers, "generate", lambda *a, **k: pytest.fail("generated"))
    request = EvaluateRequest(
        run_id=run_id, dataset_dir=directory, side="val", tasks=[TaskRef(ref="classify")]
    )
    with pytest.raises(StageError, match=r"scores a label model's predictions.*generates text"):
        evaluate(request, Context(store))


# ----------------------------------------------------------------------
# Keys, and the train stage's checkpoint
# ----------------------------------------------------------------------


def test_a_request_using_none_of_the_later_fields_dumps_and_is_keyed_as_before():
    """Pinned: these hashes are the dumps from before checkpoints were scored. docs/adr/0037"""
    evaluated = EvaluateRequest(
        run_id="20260101T000000000000-here",
        dataset_dir=Path("datasets/d/v1"),
        tasks=[TaskRef(ref="classify")],
    )
    assert content_hash(evaluated.model_dump(mode="json")) == (
        "d3d9bc14740dc665b7dabd531a94b2c01d3bf3fdcb6662f242c999ff5a2e657d"
    )
    trained = TrainStageRequest(
        dataset_dir=Path("datasets/d/v1"), model="toy.py:Toy", params={"lr": 0.1}, fresh=True
    )
    assert content_hash(trained.model_dump(mode="json")) == (
        "1549b1b9a7eb911bac0720dbd22fb17905b668f316d1513038cf181139a67c13"
    )


def test_a_request_using_a_later_field_is_another_request():
    plain = EvaluateRequest(run_id="r", dataset_dir=Path("d"))
    for changed in (
        {"checkpoints": ["epoch-2"]},
        {"settings": {"temperature": 0.5}},
        {"seed": 1},
    ):
        dumped = plain.model_copy(update=changed).model_dump(mode="json")
        assert dumped.items() >= changed.items()
        assert content_hash(dumped) != content_hash(plain.model_dump(mode="json"))
    warm = TrainStageRequest(model="m", parent="r", parent_checkpoint="epoch-1")
    assert warm.model_dump()["parent_checkpoint"] == "epoch-1"


def test_a_warm_start_continues_from_the_parents_checkpoint_it_names(store, dataset_dir):
    directory, run_id = _learner_run(store, dataset_dir)
    request = TrainStageRequest(
        dataset_dir=directory, model=LEARNER, parent=run_id, parent_checkpoint="epoch-1"
    )
    record = train(request, Context(store))
    assert record.parent_run_id == run_id
    assert record.metrics["started_at"] == 1.0


def test_a_parent_checkpoint_without_a_named_parent_is_refused(store, dataset_dir):
    directory, _ = _learner_run(store, dataset_dir)
    request = TrainStageRequest(dataset_dir=directory, model=LEARNER, parent_checkpoint="epoch-1")
    with pytest.raises(StageError, match="name the run it belongs to as parent"):
        train(request, Context(store))


def test_a_parent_checkpoint_is_refused_on_a_host_rather_than_ignored():
    request = TrainStageRequest(model=COUNTER, parent="r", parent_checkpoint="epoch-1")
    context = Context(store=None, host=Host("http://gpu", "t"), client=FakeHost)
    with pytest.raises(StageError, match="only supported for rounds trained here"):
        train(request, context)
