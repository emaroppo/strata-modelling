"""An executable specification of the model contract.

A plugin registering a model is making a promise about behaviour that
nothing else can check for it. This is the promise, written as tests a
plugin runs against its own model (see ``docs/adr/0033``):

.. code-block:: python

    from strata.modelling.plugins.conformance import ModelContract

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

Importing this pulls in pytest, so it lives behind the ``test`` extra.
"""

from typing import ClassVar

import pytest

from strata.labels import (
    BBoxSchema,
    BoxesPrediction,
    ChoicesPrediction,
    ClassificationSchema,
    SpanSchema,
    SpansPrediction,
)

from ..model import Example, Model


def _class_names(values) -> set[str]:
    """The classes a value asserts, whatever kind of value it is.

    A classification target names its classes directly; a span or a box
    carries them on each labelled thing, and a span carries a *list*, every
    entry of which is read.
    """
    names: set[str] = set()
    for value in values:
        if isinstance(value, str):
            names.add(value)
        else:
            names.update(getattr(value, "labels", None) or [value.label])
    return names


class ModelContract:
    """Subclass this in a plugin's tests and supply the two fixtures."""

    #: What a model of each task emits. Not a plugin surface: a new label
    #: type is added to ``strata.labels`` first. See ``docs/adr/0004``.
    PREDICTION_TYPES: ClassVar[dict[str, type]] = {
        "classification": ChoicesPrediction,
        "span": SpansPrediction,
        "bbox": BoxesPrediction,
    }

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
            for name in sorted(_class_names(example.target.values)):
                if name not in seen:
                    seen.append(name)
        return seen or ["alpha"]

    #: The plainest label set of each task — nothing declared on it, which
    #: is what a label set means before anyone says otherwise.
    SCHEMA_TYPES: ClassVar[dict[str, type]] = {
        "classification": ClassificationSchema,
        "span": SpanSchema,
        "bbox": BBoxSchema,
    }

    @pytest.fixture
    def schema(self, model, classes):
        """A label set this model is meant to be trained on.

        Defaults to the plainest one of its task. **Override it** where a
        model exists precisely to serve a label set that declares something
        — a single-label classifier, a tagger for overlapping spans — since
        the default is then one it is right to refuse.
        """
        task = type(model).task
        if task not in self.SCHEMA_TYPES:
            raise AssertionError(
                f"{type(model).__name__} declares task {task!r}, which names "
                f"no schema type. Known: {sorted(self.SCHEMA_TYPES)}."
            )
        return self.SCHEMA_TYPES[task](classes=list(classes))

    @pytest.fixture
    def expected_prediction(self, model) -> type:
        """What this model's task says it must emit."""
        task = type(model).task
        if task not in self.PREDICTION_TYPES:
            raise AssertionError(
                f"{type(model).__name__} declares task {task!r}, which names "
                f"no prediction type. Known: {sorted(self.PREDICTION_TYPES)}."
            )
        return self.PREDICTION_TYPES[task]

    # -- declarations ---------------------------------------------------

    def test_it_is_a_model(self, model):
        assert isinstance(model, Model)

    def test_it_declares_a_task(self, model):
        # Checked before training. docs/adr/0014
        assert isinstance(type(model).task, str) and type(model).task

    def test_it_declares_a_version(self, model):
        # A run records this, and warm-starting across a change is refused. docs/adr/0005
        assert isinstance(type(model).version, str) and type(model).version

    def test_it_accepts_the_label_set_it_is_for(self, model, schema):
        """A model may refuse a label set, but not the one it exists to serve.

        ``requires_schema`` is how a model declines a shape it cannot
        represent — overlapping spans, a region with two labels, several
        classes at once. Declining every label set of its own task is the
        thing this catches, and the task check could not.
        """
        model.requires_schema(schema)

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
        """A model may ignore it, but it has to accept it. See ``docs/adr/0031``."""
        seen = []
        metrics = model.finetune(
            examples, classes, None, lambda done, total, m: seen.append((done, total))
        )
        assert isinstance(metrics, dict)
        for done, total in seen:
            assert 1 <= done <= total

    def test_finetune_accepts_a_request_to_stop(self, model, examples, classes):
        """Asked to stop, a model may stop or carry on — but it has to finish.

        See ``docs/adr/0031``.
        """

        def stop(done, total, metrics):
            return True

        assert isinstance(model.finetune(examples, classes, None, stop), dict)

    def test_predict_accepts_a_progress_report(self, model, examples, classes):
        """A model may ignore it, but it has to accept it. See ``docs/adr/0031``."""
        model.finetune(examples, classes)
        seen = []
        outputs = model.predict(
            [e.path for e in examples], lambda done, total: seen.append((done, total))
        )
        assert len(outputs) == len(examples)
        for done, total in seen:
            assert 1 <= done <= total

    def test_predict_returns_one_output_per_path(self, model, examples, classes):
        model.finetune(examples, classes)
        paths = [e.path for e in examples]
        assert len(model.predict(paths)) == len(paths)

    def test_predict_returns_predictions(self, model, examples, classes, expected_prediction):
        model.finetune(examples, classes)
        outputs = model.predict([e.path for e in examples])
        assert all(isinstance(o, expected_prediction) for o in outputs)

    def test_predictions_stay_within_the_class_list(self, model, examples, classes):
        # A class the label set has never heard of cannot be stored. docs/adr/0014
        model.finetune(examples, classes)
        for output in model.predict([e.path for e in examples]):
            assert _class_names(output.values) <= set(classes)

    def test_predict_accepts_features(self, model, examples, classes):
        """A model may ignore them, but it has to accept them.

        Same rule as ``on_batch``. See ``docs/adr/0031``.
        """
        model.finetune(examples, classes)
        paths = [e.path for e in examples]
        outputs = model.predict(paths, None, features=[{} for _ in paths])
        assert len(outputs) == len(paths)

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
            assert _class_names(output.values) <= set(classes)


__all__ = ["ModelContract"]
