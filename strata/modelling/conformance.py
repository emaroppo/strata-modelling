"""An executable specification of the model contract.

A plugin registering a model is making a promise about behaviour that
nothing else can check for it — the registry only proves a class can be
imported. This is the promise, written as tests a plugin runs against its
own model:

.. code-block:: python

    from strata.modelling.conformance import ModelContract

    class TestMyModel(ModelContract):
        @pytest.fixture
        def model(self):
            return MyModel(epochs=1)

        @pytest.fixture
        def examples(self, tmp_path):
            return [Example(path=..., target=Choices(values=["cat"]))]

Two fixtures, and the plugin inherits the suite. What it checks is the
contract only — that a checkpoint round-trips, that predictions line up with
their inputs, that declared classes are honoured. Whether the model is any
*good* is the plugin's own business.

Importing this pulls in pytest, so it lives behind the ``test`` extra and
should only ever be imported from a test module.
"""

import pytest

from strata.labels import ChoicesPrediction

from .model import Example, Model


class ModelContract:
    """Subclass this in a plugin's tests and supply the two fixtures."""

    @pytest.fixture
    def model(self) -> Model:
        raise NotImplementedError("Supply a `model` fixture returning your model")

    @pytest.fixture
    def examples(self, tmp_path) -> list[Example]:
        raise NotImplementedError(
            "Supply an `examples` fixture returning a few Example objects "
            "whose paths point at files your model can actually read"
        )

    @pytest.fixture
    def fresh(self, model) -> Model:
        """An untrained model of the same kind, for restoring a checkpoint into.

        Default-constructed, because that is what a caller loading a run does
        — it has the reference and the recorded params, not the instance.
        Override when a model needs constructor arguments to match the one
        under test, such as pinning a device.
        """
        return type(model)()

    @pytest.fixture
    def classes(self, examples) -> list[str]:
        seen: list[str] = []
        for example in examples:
            for value in example.target.values:
                if value not in seen:
                    seen.append(value)
        return seen or ["alpha"]

    # -- declarations ---------------------------------------------------

    def test_it_is_a_model(self, model):
        assert isinstance(model, Model)

    def test_it_declares_a_task(self, model):
        # Checked before training, so a classifier pointed at a span label
        # set fails immediately rather than after a queue wait
        assert isinstance(type(model).task, str) and type(model).task

    def test_it_declares_a_version(self, model):
        # A run records this, and warm-starting across a change is refused
        assert isinstance(type(model).version, str) and type(model).version

    # -- training -------------------------------------------------------

    def test_finetune_returns_metrics(self, model, examples, classes):
        metrics = model.finetune(examples, classes)
        assert isinstance(metrics, dict)
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_finetune_accepts_a_validation_set(self, model, examples, classes):
        assert isinstance(model.finetune(examples[:-1], classes, examples[-1:]), dict)

    def test_finetune_accepts_no_validation_set(self, model, examples, classes):
        # val is optional: a first round may have too little data to hold any
        # back, and that must not be a crash
        assert isinstance(model.finetune(examples, classes, None), dict)

    # -- prediction -----------------------------------------------------

    def test_finetune_accepts_a_progress_report(self, model, examples, classes):
        """A model may ignore it, but it has to accept it.

        The caller is often on the other end of a network and cannot see
        training happen. Reporting is optional — silence means no news, not
        a stall — but refusing the argument fails the round rather than the
        contract, and does so ten minutes in.
        """
        seen = []
        metrics = model.finetune(
            examples, classes, None, lambda done, total, m: seen.append((done, total))
        )
        assert isinstance(metrics, dict)
        for done, total in seen:
            assert 1 <= done <= total

    def test_predict_returns_one_output_per_path(self, model, examples, classes):
        model.finetune(examples, classes)
        paths = [e.path for e in examples]
        assert len(model.predict(paths)) == len(paths)

    def test_predict_returns_predictions(self, model, examples, classes):
        model.finetune(examples, classes)
        outputs = model.predict([e.path for e in examples])
        assert all(isinstance(o, ChoicesPrediction) for o in outputs)

    def test_predictions_stay_within_the_class_list(self, model, examples, classes):
        # A class the label set has never heard of cannot be stored, so
        # inventing one turns into a validation failure much later
        model.finetune(examples, classes)
        for output in model.predict([e.path for e in examples]):
            assert set(output.values) <= set(classes)

    def test_predict_on_nothing_returns_nothing(self, model, examples, classes):
        model.finetune(examples, classes)
        assert model.predict([]) == []

    # -- checkpoints ----------------------------------------------------

    def test_save_writes_something(self, model, examples, classes, tmp_path):
        model.finetune(examples, classes)
        checkpoint = tmp_path / "checkpoint.pt"
        model.save(checkpoint)
        assert checkpoint.exists() and checkpoint.stat().st_size > 0

    def test_a_checkpoint_round_trips(self, model, fresh, examples, classes, tmp_path):
        model.finetune(examples, classes)
        paths = [e.path for e in examples]
        before = model.predict(paths)

        checkpoint = tmp_path / "checkpoint.pt"
        model.save(checkpoint)

        restored = fresh
        restored.load(checkpoint)

        # The whole point of a checkpoint: the same inputs give the same
        # answers, in a fresh process that never saw the training data
        assert [o.values for o in restored.predict(paths)] == [o.values for o in before]

    def test_a_restored_model_keeps_its_classes(self, model, fresh, examples, classes, tmp_path):
        model.finetune(examples, classes)
        checkpoint = tmp_path / "checkpoint.pt"
        model.save(checkpoint)

        restored = fresh
        restored.load(checkpoint)
        for output in restored.predict([e.path for e in examples]):
            assert set(output.values) <= set(classes)


__all__ = ["ModelContract"]
