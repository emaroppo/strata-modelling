"""Training and prediction through the handler both transports call."""

import json

import pytest
from counting_model import COUNTER, COUNTING_MODEL

from strata.labels import MANIFEST_FORMAT
from strata.modelling import (
    ModelError,
    PredictRequest,
    TrainingError,
    TrainRequest,
    predict,
    train,
)

#: Bolted onto the generated model so it declines a label set whose shape
#: it cannot represent — the same thing the span tagger does about BIO.
REFUSES_OVERLAPS = """

def _refuse(self, schema):
    if getattr(schema, "overlapping", False):
        raise ValueError("each token gets one tag")


CountingModel.requires_schema = _refuse
"""

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def test_training_records_a_run(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    assert run.id
    assert store.get(run.id) == run


def test_the_run_records_its_experiment_and_what_it_saw(store, dataset_dir):
    asked = "e" * 64
    request = TrainRequest(
        dataset_dir=dataset_dir(n_train=6, n_val=2), model=COUNTER, experiment_id=asked
    )
    run = train(request, store)
    assert store.get(run.id).experiment_id == asked
    assert [r.id for r in store.for_experiment(asked)] == [run.id]
    # Every sample of the manifest, by side, so the split as realised is
    # asked of the run rather than of a directory
    saw = store.saw(run.id)
    assert {side: len(sums) for side, sums in saw.items()} == {"train": 6, "val": 2, "holdout": 0}
    # A round nobody orchestrated has no experiment, and says so
    alone = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    assert store.get(alone.id).experiment_id is None
    assert store.for_experiment(asked) == [store.get(run.id)]


def test_the_run_carries_its_lineage(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(version=3), model=COUNTER), store)
    # These three resolve back to the exact samples and annotations behind
    # the checkpoint
    assert (run.dataset, run.dataset_version, run.label_set) == ("d", 3, "presence")


def test_the_run_records_the_class_list_as_trained(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    # The label set may gain classes later; this is what the checkpoint encodes
    assert run.classes == ["cat", "dog"]


def test_the_run_records_the_model_version(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    assert run.model_version == "1"


def test_the_split_from_the_manifest_reaches_the_model(store, dataset_dir):
    run = train(
        TrainRequest(dataset_dir=dataset_dir(n_train=7, n_val=3), model=COUNTER), store
    )
    assert run.metrics["n_train"] == 7


def test_a_manifest_written_without_a_catalog_trains(store, tmp_path):
    """The smallest manifest a producer outside strata could honestly write.

    No catalog id, no sample ids, no version, no split ratios — nothing only
    a strata catalog could know. A dataset folder is the portable unit, and
    requiring any of those would make whoever wrote it invent them.
    """
    root = tmp_path / "outside"
    (root / "files").mkdir(parents=True)
    (root / COUNTER.split(":")[0]).write_text(COUNTING_MODEL)
    samples = []
    for i, split in enumerate(["train", "train", "train", "val"]):
        (root / "files" / f"{i}.jpg").write_bytes(f"image {i}".encode())
        samples.append(
            {
                "checksum": f"{i:064x}",
                "path": f"files/{i}.jpg",
                "split": split,
                "value": {"kind": "choices", "values": ["cat"]},
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "dataset": "outside",
                "label_set": "presence",
                "label_schema": {"task": "classification", "classes": ["cat", "dog"]},
                "samples": samples,
            }
        )
    )

    run = train(TrainRequest(dataset_dir=root, model=COUNTER), store)

    assert (run.metrics["n_train"], run.metrics["n_val"]) == (3, 1)
    assert (run.dataset_version, run.catalog_id) == (None, None)


def test_a_holdout_never_reaches_the_model(store, dataset_dir):
    """Not as training data, and not as validation either.

    A holdout exists to measure what search and selection never saw; a
    model scored on it during training has seen it.
    """
    root = dataset_dir(n_train=6, n_val=3)
    manifest = json.loads((root / "manifest.json").read_text())
    kept_back = [s for s in manifest["samples"] if s["split"] == "train"][:2] + [
        s for s in manifest["samples"] if s["split"] == "val"
    ][:1]
    for sample in kept_back:
        sample["split"] = "holdout"
    (root / "manifest.json").write_text(json.dumps(manifest))

    run = train(TrainRequest(dataset_dir=root, model=COUNTER), store)
    assert run.metrics["n_train"] == 4
    assert run.metrics["n_val"] == 2


def test_skipped_samples_are_not_training_data(store, dataset_dir):
    run = train(
        TrainRequest(dataset_dir=dataset_dir(n_train=5, n_val=2, n_skipped=4), model=COUNTER),
        store,
    )
    assert run.metrics["n_train"] == 5


def test_params_reach_the_constructor(store, dataset_dir):
    run = train(
        TrainRequest(dataset_dir=dataset_dir(), model=COUNTER, params={"bias": 0.9}), store
    )
    assert run.metrics["accuracy"] == pytest.approx(0.9)
    assert run.params == {"bias": 0.9}


def test_a_checkpoint_is_written_and_recorded(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    assert run.checkpoint.exists()
    assert store.get(run.id).checkpoint == run.checkpoint


def test_a_missing_manifest_is_an_error(store, tmp_path):
    with pytest.raises(TrainingError, match="No manifest.json"):
        train(TrainRequest(dataset_dir=tmp_path, model=COUNTER), store)


def test_a_manifest_this_release_cannot_read_is_refused(store, dataset_dir):
    """The core holds a directory, not a catalog: it cannot fetch a fresh copy, so it says why."""
    root = dataset_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["format"] = 2
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TrainingError, match="format 2"):
        train(TrainRequest(dataset_dir=root, model=COUNTER), store)


def test_a_dataset_with_no_training_samples_is_an_error(store, dataset_dir):
    with pytest.raises(TrainingError, match="no training samples"):
        train(TrainRequest(dataset_dir=dataset_dir(n_train=0, n_val=2), model=COUNTER), store)


def test_a_model_for_another_task_is_refused(store, dataset_dir):
    root = dataset_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["label_schema"]["task"] = "classification"
    (root / "counter.py").write_text(
        (root / "counter.py").read_text().replace('task = "classification"', 'task = "span"')
    )
    with pytest.raises(TrainingError, match="handles 'span'"):
        train(TrainRequest(dataset_dir=root, model=COUNTER), store)


def test_a_label_set_shaped_wrong_for_the_model_is_refused(store, dataset_dir):
    """The finer check the task check cannot make.

    Right task, wrong shape: the model handles spans, and this label set
    declares overlapping ones. Caught here it is a refusal; caught nowhere,
    the model trains on a projection of the data and reports a number for
    the projection.
    """
    root = dataset_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["label_schema"] = {
        "task": "span",
        "classes": ["cat", "dog"],
        "overlapping": True,
    }
    for sample in manifest["samples"]:
        if sample["value"] is not None:
            sample["value"] = {"kind": "spans", "values": []}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "counter.py").write_text(
        (root / "counter.py").read_text().replace(
            'task = "classification"', 'task = "span"'
        )
        + REFUSES_OVERLAPS
    )
    with pytest.raises(TrainingError, match="cannot be trained on this label set"):
        train(TrainRequest(dataset_dir=root, model=COUNTER), store)


def test_a_model_with_nothing_to_say_about_the_schema_trains(store, dataset_dir):
    # The default accepts anything, so a plugin written before the hook
    # existed keeps working without knowing about it
    assert train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store).id


# ----------------------------------------------------------------------
# Warm starting
# ----------------------------------------------------------------------


def test_a_warm_start_loads_the_parent_checkpoint(store, dataset_dir):
    first = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    second = train(
        TrainRequest(
            dataset_dir=dataset_dir(version=2), model=COUNTER, parent_run_id=first.id
        ),
        store,
    )
    assert second.parent_run_id == first.id


def test_the_chain_walks_back_to_the_cold_start(store, dataset_dir):
    first = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    second = train(
        TrainRequest(dataset_dir=dataset_dir(version=2), model=COUNTER, parent_run_id=first.id),
        store,
    )
    third = train(
        TrainRequest(dataset_dir=dataset_dir(version=3), model=COUNTER, parent_run_id=second.id),
        store,
    )
    # Warm-started metrics only mean something against the run before them
    assert [r.id for r in store.chain(third.id)] == [first.id, second.id, third.id]


def test_a_cold_start_has_no_parent(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    assert run.parent_run_id is None


def test_continuing_from_a_missing_run_is_an_error(store, dataset_dir):
    with pytest.raises(TrainingError, match="No run with id 99"):
        train(
            TrainRequest(dataset_dir=dataset_dir(), model=COUNTER, parent_run_id="99"), store
        )


def test_appending_a_class_still_warm_starts(store, dataset_dir):
    first = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    grown = dataset_dir(version=2, classes=("cat", "dog", "bird"))
    second = train(
        TrainRequest(dataset_dir=grown, model=COUNTER, parent_run_id=first.id), store
    )
    assert second.classes == ["cat", "dog", "bird"]


def test_reordering_classes_refuses_to_warm_start(store, dataset_dir):
    first = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    reordered = dataset_dir(version=2, classes=("dog", "cat"))
    # A checkpoint maps output neurons to the class list by position, so this
    # corrupts silently rather than failing if it is allowed through
    with pytest.raises(TrainingError, match="append-only"):
        train(TrainRequest(dataset_dir=reordered, model=COUNTER, parent_run_id=first.id), store)


def test_removing_a_class_refuses_to_warm_start(store, dataset_dir):
    first = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    shrunk = dataset_dir(version=2, classes=("cat",))
    with pytest.raises(TrainingError, match="append-only"):
        train(TrainRequest(dataset_dir=shrunk, model=COUNTER, parent_run_id=first.id), store)


def test_a_model_version_change_refuses_to_warm_start(store, dataset_dir):
    first = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    bumped = dataset_dir(version=2)
    (bumped / "counter.py").write_text(
        (bumped / "counter.py").read_text().replace('version = "1"', 'version = "2"')
    )
    with pytest.raises(TrainingError, match="version 1"):
        train(TrainRequest(dataset_dir=bumped, model=COUNTER, parent_run_id=first.id), store)


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------


def test_prediction_returns_one_output_per_path(store, dataset_dir):
    root = dataset_dir()
    run = train(TrainRequest(dataset_dir=root, model=COUNTER), store)
    paths = sorted((root / "files").glob("*.jpg"))[:4]
    assert len(predict(PredictRequest(run_id=run.id, paths=paths), store)) == 4


def test_predictions_keep_their_paths(store, dataset_dir):
    root = dataset_dir()
    run = train(TrainRequest(dataset_dir=root, model=COUNTER), store)
    paths = sorted((root / "files").glob("*.jpg"))[:3]
    assert [p.path for p in predict(PredictRequest(run_id=run.id, paths=paths), store)] == paths


def test_prediction_restores_the_classes_from_the_checkpoint(store, dataset_dir):
    root = dataset_dir()
    run = train(TrainRequest(dataset_dir=root, model=COUNTER), store)
    [first] = predict(
        PredictRequest(run_id=run.id, paths=[sorted((root / "files").glob("*.jpg"))[0]]), store
    )
    assert first.value.values == ["cat"]


def test_predicting_from_an_unknown_run_is_an_error(store):
    with pytest.raises(TrainingError, match="No run with id 42"):
        predict(PredictRequest(run_id="42", paths=[]), store)


def test_predicting_without_a_checkpoint_is_an_error(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    run.checkpoint.unlink()
    with pytest.raises(TrainingError, match="no checkpoint"):
        predict(PredictRequest(run_id=run.id, paths=[]), store)


def test_an_unresolvable_model_is_an_error(store, dataset_dir):
    with pytest.raises(ModelError):
        train(TrainRequest(dataset_dir=dataset_dir(), model="nope.py:Missing"), store)


def test_a_bad_parameter_names_itself(store, dataset_dir):
    # In process this is a TypeError from inside the constructor; over a wire
    # it would be a 500 with a traceback. One error, the same either side.
    with pytest.raises(TrainingError, match="will not accept these parameters"):
        train(
            TrainRequest(dataset_dir=dataset_dir(), model=COUNTER, params={"nope": 1}),
            store,
        )


def test_the_error_lists_what_the_model_does_take(store, dataset_dir):
    with pytest.raises(TrainingError, match="It takes: bias"):
        train(
            TrainRequest(dataset_dir=dataset_dir(), model=COUNTER, params={"nope": 1}),
            store,
        )


def test_a_model_needing_an_undeclared_class_is_refused(store, dataset_dir):
    # A model with an implicit negative predicts a token the label set never
    # declared. That is legal here and rejected by whatever displays it —
    # silently, and worst on exactly the samples worth reviewing.
    root = dataset_dir()
    (root / "counter.py").write_text(
        (root / "counter.py").read_text().replace(
            '    version = "1"', '    version = "1"\n    requires_classes = ("none",)'
        )
    )
    with pytest.raises(TrainingError, match="does not declare"):
        train(TrainRequest(dataset_dir=root, model=COUNTER), store)


def test_a_declared_requirement_is_accepted(store, dataset_dir):
    root = dataset_dir(classes=("cat", "none"))
    (root / "counter.py").write_text(
        (root / "counter.py").read_text().replace(
            '    version = "1"', '    version = "1"\n    requires_classes = ("none",)'
        )
    )
    assert train(TrainRequest(dataset_dir=root, model=COUNTER), store).id


def test_continuing_from_another_model_is_refused(store, dataset_dir):
    first = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    other = dataset_dir(version=2)
    (other / "counter.py").write_text(
        (other / "counter.py").read_text().replace("class CountingModel", "class Other")
    )
    with pytest.raises(TrainingError, match="is not a warm start"):
        train(
            TrainRequest(
                dataset_dir=other, model="counter.py:Other", parent_run_id=first.id
            ),
            store,
        )


def test_the_same_model_under_another_name_still_continues(store, dataset_dir):
    # One class is recorded under whatever spelling the caller used, so
    # comparing references rather than classes would refuse a rename
    root = dataset_dir()
    first = train(TrainRequest(dataset_dir=root, model=COUNTER), store)
    absolute_ref = f"{root / 'counter.py'}:CountingModel"
    second = train(
        TrainRequest(
            dataset_dir=dataset_dir(version=2), model=absolute_ref, parent_run_id=first.id
        ),
        store,
    )
    assert second.parent_run_id == first.id


def test_an_unresolvable_parent_reference_does_not_block(store, dataset_dir):
    # The ordinary state of a run imported from an older layout: refusing on
    # it would make history unusable to say nothing about it
    first = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    with store.engine.begin() as conn:
        from strata.modelling.store import tables as t

        conn.execute(
            t.run.update().where(t.run.c.id == first.id).values(model="gone.away:Model")
        )
    second = train(
        TrainRequest(
            dataset_dir=dataset_dir(version=2), model=COUNTER, parent_run_id=first.id
        ),
        store,
    )
    assert second.parent_run_id == first.id
