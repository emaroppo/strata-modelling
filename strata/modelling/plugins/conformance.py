"""An executable specification of the model contract, as tests a plugin runs against its own model.

Subclass :class:`ModelContract` and supply the ``model`` and ``examples``
fixtures; imports pytest, so it lives behind the ``test`` extra. See ``docs/adr/0033``.
"""

from typing import ClassVar

import pytest

from strata.contracts import (
    BBoxSchema,
    BoxesPrediction,
    ChoicesPrediction,
    ClassificationSchema,
    SpanSchema,
    SpansPrediction,
)

from ..model import Example, LabelModel


def _class_names(values) -> set[str]:
    """The classes a value asserts, whatever kind of value it is."""
    names: set[str] = set()
    for value in values:
        if isinstance(value, str):
            names.add(value)
        else:
            names.update(getattr(value, "labels", None) or [value.label])
    return names


class ModelContract:
    """Subclass this in a plugin's tests and supply the two fixtures."""

    #: What a model of each label type emits; a new label type is added to ``strata.contracts``
    #: first (``docs/adr/0004``).
    PREDICTION_TYPES: ClassVar[dict[str, type]] = {
        "classification": ChoicesPrediction,
        "span": SpansPrediction,
        "bbox": BoxesPrediction,
    }

    @pytest.fixture
    def model(self) -> LabelModel:
        raise NotImplementedError("Supply a `model` fixture returning your model")

    @pytest.fixture
    def examples(self, tmp_path) -> list[Example]:
        raise NotImplementedError(
            "Supply an `examples` fixture returning a few Example objects "
            "whose paths point at files your model can actually read"
        )

    @pytest.fixture
    def fresh(self, model) -> LabelModel:
        """An untrained, default-constructed model of the same kind, for restoring a checkpoint
        into.

        Override when a model needs constructor arguments to match the one under test.
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

    #: The plainest label set of each label type, nothing declared on it.
    SCHEMA_TYPES: ClassVar[dict[str, type]] = {
        "classification": ClassificationSchema,
        "span": SpanSchema,
        "bbox": BBoxSchema,
    }

    @pytest.fixture
    def schema(self, model, classes):
        """A label set this model is meant to be trained on: the plainest of its label type.

        Override it where the model exists to serve a label set that declares something.
        """
        label_type = type(model).label_type
        if label_type not in self.SCHEMA_TYPES:
            raise AssertionError(
                f"{type(model).__name__} declares label type {label_type!r}, which "
                f"names no schema type. Known: {sorted(self.SCHEMA_TYPES)}."
            )
        return self.SCHEMA_TYPES[label_type](classes=list(classes))

    @pytest.fixture
    def expected_prediction(self, model) -> type:
        """What this model's label type says it must emit."""
        label_type = type(model).label_type
        if label_type not in self.PREDICTION_TYPES:
            raise AssertionError(
                f"{type(model).__name__} declares label type {label_type!r}, which "
                f"names no prediction type. Known: {sorted(self.PREDICTION_TYPES)}."
            )
        return self.PREDICTION_TYPES[label_type]

    # -- declarations ---------------------------------------------------

    def test_it_is_a_model(self, model):
        assert isinstance(model, LabelModel)

    def test_it_declares_a_label_type(self, model):
        assert isinstance(type(model).label_type, str) and type(model).label_type

    def test_it_declares_a_version(self, model):
        assert isinstance(type(model).version, str) and type(model).version

    def test_it_accepts_the_label_set_it_is_for(self, model, schema):
        """A model may refuse a label set, but not the one it exists to serve."""
        model.requires_schema(schema)

    # -- training -------------------------------------------------------

    def test_finetune_returns_metrics(self, model, examples, classes):
        metrics = model.finetune(examples, classes)
        assert isinstance(metrics, dict)
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_finetune_accepts_a_validation_set(self, model, examples, classes):
        assert isinstance(model.finetune(examples[:-1], classes, examples[-1:]), dict)

    def test_finetune_accepts_no_validation_set(self, model, examples, classes):
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
        """Asked to stop, a model may stop or carry on, but it has to finish (``docs/adr/0031``)."""

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
        # docs/adr/0014
        model.finetune(examples, classes)
        for output in model.predict([e.path for e in examples]):
            assert _class_names(output.values) <= set(classes)

    def test_predict_accepts_features(self, model, examples, classes):
        """A model may ignore them, but it has to accept them. See ``docs/adr/0031``."""
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
