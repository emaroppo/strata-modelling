"""A model that makes media: trained on samples of its type, and asked to generate.

A media model is checked against the samples rather than a label set,
handed examples whose file is the target, and asked for outputs it returns
rather than writes. See ``docs/adr/0054``.
"""

import json
import re

import pytest
from counting_model import COUNTER
from media_model import ECHO

from strata.common.canonical import content_hash
from strata.contracts import Manifest
from strata.modelling import (
    GenerateRequest,
    GenerationInput,
    PredictRequest,
    TrainingError,
    TrainRequest,
    examples,
    generate,
    predict,
    train,
)

# ----------------------------------------------------------------------
# What is checked before the round
# ----------------------------------------------------------------------


def test_a_media_model_trains_on_a_subtype_of_its_type(store, sft_dir):
    run = train(TrainRequest(dataset_dir=sft_dir(), model=ECHO), store)
    assert (run.metrics["n_train"], run.metrics["n_val"]) == (4, 2)
    # Learns from the samples, not the verdicts that chose them
    assert run.classes == []


def test_a_media_model_trains_on_its_own_type(store, sft_dir):
    assert train(TrainRequest(dataset_dir=sft_dir(types=("text/sft",) * 4), model=ECHO), store).id


def test_samples_of_another_type_are_refused_by_name(store, sft_dir):
    # text/sftx shares a prefix and is not a subtype: the rule is a path
    root = sft_dir(types=("text/sft", "text/email", "text/sftx", "text/sft"))
    with pytest.raises(TrainingError, match=re.escape("holds text/email, text/sftx")):
        train(TrainRequest(dataset_dir=root, model=ECHO), store)


def test_samples_that_do_not_say_their_type_are_refused(store, sft_dir):
    root = sft_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["samples"][0]["type"] = None
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TrainingError, match="Materialise the dataset again"):
        train(TrainRequest(dataset_dir=root, model=ECHO), store)


def test_a_media_model_naming_no_type_is_refused(store, sft_dir):
    root = sft_dir()
    (root / "echo.py").write_text(
        (root / "echo.py").read_text().replace("trains_on = SFTTextExample", "")
    )
    with pytest.raises(TrainingError, match="names no sample type in `trains_on`"):
        train(TrainRequest(dataset_dir=root, model=ECHO), store)


def test_a_media_model_still_needs_its_features(store, sft_dir):
    root = sft_dir()
    (root / "echo.py").write_text(
        (root / "echo.py")
        .read_text()
        .replace('    version = "1"', '    version = "1"\n    requires_features = ("sender",)')
    )
    with pytest.raises(TrainingError, match=re.escape("needs feature(s) sender")):
        train(TrainRequest(dataset_dir=root, model=ECHO), store)


def test_a_media_models_examples_carry_no_target(sft_dir):
    root = sft_dir()
    manifest = Manifest.model_validate_json((root / "manifest.json").read_text())
    train_examples, val_examples = examples(root, manifest, targets=False)
    # The file is what it learns to make, and the features what it is told
    assert all(e.target is None for e in train_examples + val_examples)
    assert train_examples[0].features == {"subject": "subject 0"}
    assert train_examples[0].path.read_text() == "body 0"


def test_the_model_is_handed_no_targets_in_training(store, sft_dir):
    run = train(TrainRequest(dataset_dir=sft_dir(), model=ECHO, params={"epochs": 1}), store)
    seen = json.loads(run.checkpoint.read_text())["seen"]
    assert seen["targets"] == [None] * 4
    assert seen["features"][0] == {"subject": "subject 0"}


# ----------------------------------------------------------------------
# Generating
# ----------------------------------------------------------------------


def _ask(run_id, n=3, **extra) -> GenerateRequest:
    return GenerateRequest(
        run_id=run_id,
        inputs=[GenerationInput(values={"subject": f"s{i}"}) for i in range(n)],
        settings=[{"temperature": 0.1 * i} for i in range(n)],
        seeds=[10 + i for i in range(n)],
        **extra,
    )


def test_generate_returns_one_output_per_input_with_its_settings_digest_and_seed(store, sft_dir):
    run = train(TrainRequest(dataset_dir=sft_dir(), model=ECHO), store)
    asked = _ask(run.id)

    outputs = generate(asked, store)

    assert [json.loads(o.data)["values"] for o in outputs] == [i.values for i in asked.inputs]
    assert [o.seed for o in outputs] == [10, 11, 12]
    # Over the canonical form, so the digest is the one any other writer of
    # the same settings computes
    assert [o.settings_digest for o in outputs] == [content_hash(s) for s in asked.settings]
    assert [o.ended for o in outputs] == [True, False, True]


def test_generate_uses_the_runs_last_checkpoint_unless_one_is_named(store, sft_dir):
    run = train(TrainRequest(dataset_dir=sft_dir(), model=ECHO), store)

    [last] = generate(_ask(run.id, n=1), store)
    [second] = generate(_ask(run.id, n=1, checkpoint="epoch-2"), store)

    assert (json.loads(last.data)["epoch"], last.checkpoint) == (3, "epoch-3")
    assert (json.loads(second.data)["epoch"], second.checkpoint) == (2, "epoch-2")
    # Named by digest as well as by name, for the generation record (0053)
    assert second.checkpoint_digest == run.checkpoint_named("epoch-2").digest


def test_an_input_is_given_its_samples_by_role(store, sft_dir, tmp_path):
    run = train(TrainRequest(dataset_dir=sft_dir(), model=ECHO), store)
    source = tmp_path / "to-summarise.txt"
    source.write_text("the long text")

    [out] = generate(
        GenerateRequest(
            run_id=run.id,
            inputs=[GenerationInput(values={}, samples={"source": [source]})],
            settings=[{}],
            seeds=[0],
        ),
        store,
    )

    assert json.loads(out.data)["samples"] == {"source": ["the long text"]}


def test_inputs_settings_and_seeds_must_line_up():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="positional"):
        GenerateRequest(run_id="r", inputs=[GenerationInput()], settings=[], seeds=[1])


def test_generating_from_a_label_models_run_is_refused(store, dataset_dir):
    run = train(TrainRequest(dataset_dir=dataset_dir(), model=COUNTER), store)
    with pytest.raises(TrainingError, match="a label model: it predicts"):
        generate(_ask(run.id), store)


def test_predicting_with_a_media_models_run_is_refused(store, sft_dir):
    run = train(TrainRequest(dataset_dir=sft_dir(), model=ECHO), store)
    with pytest.raises(TrainingError, match="a media model: it generates"):
        predict(PredictRequest(run_id=run.id, paths=[]), store)


def test_naming_a_checkpoint_the_run_never_kept_is_refused(store, sft_dir):
    run = train(TrainRequest(dataset_dir=sft_dir(), model=ECHO), store)
    with pytest.raises(TrainingError, match="It kept: epoch-1, epoch-2, epoch-3"):
        generate(_ask(run.id, checkpoint="epoch-9"), store)


def test_a_run_records_the_params_the_model_says_it_used(store, sft_dir):
    """Defaults included, so a changed default never changes what a run means (0054)."""
    root = sft_dir()
    (root / "echo.py").write_text(
        (root / "echo.py")
        .read_text()
        .replace(
            "    def save(self, path):",
            '    def resolved_params(self):\n        return {"epochs": self.epochs}\n\n'
            "    def save(self, path):",
        )
    )

    run = train(TrainRequest(dataset_dir=root, model=ECHO), store)

    assert run.params == {"epochs": 3}
    assert store.get(run.id).params == {"epochs": 3}
    # And the model is rebuilt from them to generate
    assert generate(_ask(run.id, n=1), store)[0].checkpoint == "epoch-3"


def test_a_model_that_says_nothing_records_the_params_asked_for(store, sft_dir):
    run = train(TrainRequest(dataset_dir=sft_dir(), model=ECHO, params={"epochs": 2}), store)
    assert run.params == {"epochs": 2}
