"""The image baselines' task hooks, transform and warm start.

Needs the image extra, but not a GPU and not the network: nothing here
builds the real backbone, which would fetch pretrained weights.
"""

import pytest

pytest.importorskip("timm", reason="needs the image extra")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

from strata.modelling.baselines.classifier import (  # noqa: E402
    MulticlassClassifier,
    MultiLabelClassifier,
    PresenceClassifier,
    _LetterboxSquash,
)

CLASSES = ["cat", "dog", "bird"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


def classifier(cls, classes=CLASSES):
    model = cls(device="cpu")
    model.classes = list(classes)
    return model


# ----------------------------------------------------------------------
# Target encoding
# ----------------------------------------------------------------------


def test_multilabel_encodes_a_multi_hot_vector():
    encoded = MultiLabelClassifier._encode_target(["cat", "bird"], CLASS_TO_IDX)
    assert encoded.tolist() == [1.0, 0.0, 1.0]


def test_multilabel_ignores_labels_outside_the_head():
    # PresenceClassifier depends on this: "none" has no neuron and has to
    # train as an all-zeros target rather than raising
    encoded = MultiLabelClassifier._encode_target(["cat", "none"], CLASS_TO_IDX)
    assert encoded.tolist() == [1.0, 0.0, 0.0]


def test_multilabel_encodes_an_empty_target_as_all_zeros():
    assert MultiLabelClassifier._encode_target([], CLASS_TO_IDX).tolist() == [0.0, 0.0, 0.0]


def test_multiclass_encodes_the_first_label_as_an_index():
    encoded = MulticlassClassifier._encode_target(["dog"], CLASS_TO_IDX)
    assert encoded.dtype == torch.long
    assert encoded.item() == 1


def test_multiclass_takes_the_first_label_when_the_data_has_extras():
    assert MulticlassClassifier._encode_target(["bird", "cat"], CLASS_TO_IDX).item() == 2


def test_multiclass_encodes_an_empty_target_as_class_zero():
    """Index 0 is a real class, not a null: an unlabelled sample reaching
    here trains as the first class. Callers are expected not to."""
    assert MulticlassClassifier._encode_target([], CLASS_TO_IDX).item() == 0


# ----------------------------------------------------------------------
# Accuracy counting
# ----------------------------------------------------------------------


def test_multilabel_counts_only_an_exact_set_match():
    logits = torch.tensor([[3.0, -3.0, 3.0], [3.0, -3.0, -3.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    # The second row gets one of two classes right and still counts as wrong
    assert MultiLabelClassifier._count_correct(logits, targets) == 1


def test_multiclass_counts_the_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2], [4.0, 0.1, 0.2]])
    targets = torch.tensor([1, 2])
    assert MulticlassClassifier._count_correct(logits, targets) == 1


# ----------------------------------------------------------------------
# Decoding a prediction
# ----------------------------------------------------------------------


def test_multilabel_returns_everything_over_the_threshold_most_confident_first():
    output = classifier(MultiLabelClassifier)._to_output(torch.tensor([0.7, 0.2, 0.9]))
    assert output.values == ["bird", "cat"]
    assert output.confidences == [0.9, 0.7]


def test_multilabel_falls_back_to_the_argmax_when_nothing_clears():
    output = classifier(MultiLabelClassifier)._to_output(torch.tensor([0.3, 0.2, 0.1]))
    assert output.values == ["cat"]
    assert output.confidences == [0.3]


def test_multiclass_returns_exactly_one_label():
    output = classifier(MulticlassClassifier)._to_output(torch.tensor([0.2, 0.5, 0.3]))
    assert output.values == ["dog"]
    assert output.confidences == [0.5]


def test_presence_reports_the_negative_class_when_nothing_clears():
    model = classifier(PresenceClassifier, ["cat", "dog"])
    output = model._to_output(torch.tensor([0.3, 0.1]))
    assert output.values == [PresenceClassifier.NEGATIVE_LABEL]
    # Confidence in "nothing here" is how far the best guess fell short
    assert output.confidences == [pytest.approx(0.7)]


def test_presence_never_reports_the_negative_class_beside_a_positive():
    model = classifier(PresenceClassifier, ["cat", "dog"])
    output = model._to_output(torch.tensor([0.9, 0.6]))
    assert output.values == ["cat", "dog"]
    assert PresenceClassifier.NEGATIVE_LABEL not in output.values


def test_presence_drops_the_negative_class_from_the_head():
    model = PresenceClassifier(device="cpu")
    assert model._effective_classes(["cat", "none", "dog"]) == ["cat", "dog"]


# ----------------------------------------------------------------------
# The transform
# ----------------------------------------------------------------------


@pytest.mark.parametrize("size", [(320, 180), (180, 320), (256, 256), (100, 33)])
def test_letterbox_always_returns_the_requested_square(size):
    assert _LetterboxSquash(224)(Image.new("RGB", size)).size == (224, 224)


def test_a_square_image_is_left_undistorted():
    squash = _LetterboxSquash(64)
    result = squash(Image.new("RGB", (256, 256), "white"))
    # No padding: every pixel is content
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((63, 63)) == (255, 255, 255)


def test_widescreen_content_fills_the_documented_share_of_the_square():
    squash = _LetterboxSquash(1000, max_distortion=1.4)
    result = squash(Image.new("RGB", (1600, 900), "white"))
    filled = sum(1 for y in range(1000) if result.getpixel((500, y)) != (0, 0, 0))
    # The docstring's claim: ~79% for 16:9 at max_distortion 1.4
    assert filled / 1000 == pytest.approx(0.79, abs=0.01)


def test_distortion_never_exceeds_the_maximum():
    squash = _LetterboxSquash(1000, max_distortion=1.4)
    result = squash(Image.new("RGB", (4000, 500), "white"))
    filled = sum(1 for y in range(1000) if result.getpixel((500, y)) != (0, 0, 0))
    # An 8:1 image is squashed by exactly 1.4 and letterboxed for the rest
    assert (1000 / filled) == pytest.approx(8.0 / 1.4, rel=0.01)


def test_the_content_is_centred():
    squash = _LetterboxSquash(100, max_distortion=1.0)
    result = squash(Image.new("RGB", (400, 100), "white"))
    column = [result.getpixel((50, y)) != (0, 0, 0) for y in range(100)]
    above = column.index(True)
    below = len(column) - 1 - column[::-1].index(True)
    assert above == pytest.approx(100 - 1 - below, abs=1)


# ----------------------------------------------------------------------
# Warm start
# ----------------------------------------------------------------------


class _TinyBackbone(nn.Module):
    """Stands in for the timm model, with the two methods _expand_head calls.

    The real one downloads pretrained weights, which a unit test should not.
    """

    def __init__(self, num_classes: int, features: int = 4):
        super().__init__()
        self.features = features
        self.head = nn.Linear(features, num_classes)

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        self.head = nn.Linear(self.features, num_classes)

    def forward(self, x):
        return self.head(x)


class StubClassifier(MultiLabelClassifier):
    def _build_backbone(self, num_classes: int) -> nn.Module:
        return _TinyBackbone(num_classes).to(self.device)


def stub(classes: list[str]) -> StubClassifier:
    model = StubClassifier(device="cpu")
    model._backbone = model._build_backbone(len(classes))
    model.classes = list(classes)
    return model


def test_expanding_the_head_keeps_the_rows_already_learned():
    model = stub(["cat", "dog"])
    before = model._backbone.get_classifier()
    weight, bias = before.weight.data.clone(), before.bias.data.clone()

    model._expand_head(3)

    after = model._backbone.get_classifier()
    assert after.weight.shape[0] == 3
    # Adding a class must not cost the rounds already trained
    assert torch.equal(after.weight.data[:2], weight)
    assert torch.equal(after.bias.data[:2], bias)


def test_an_unchanged_class_list_does_not_touch_the_backbone():
    model = stub(["cat", "dog"])
    backbone = model._backbone
    model._prepare_backbone(["cat", "dog"])
    assert model._backbone is backbone


def test_appending_a_class_grows_the_head_in_place():
    model = stub(["cat", "dog"])
    backbone = model._backbone
    weight = backbone.get_classifier().weight.data.clone()

    model._prepare_backbone(["cat", "dog", "bird"])

    assert model._backbone is backbone
    assert torch.equal(model._backbone.get_classifier().weight.data[:2], weight)


@pytest.mark.parametrize(
    "changed",
    [
        ["dog", "cat"],  # reordered: every neuron would mean something else
        ["cat"],  # removed
        ["bird", "cat", "dog"],  # inserted before the existing ones
    ],
)
def test_an_incompatible_class_list_rebuilds_from_pretrained(changed):
    model = stub(["cat", "dog"])
    backbone = model._backbone
    model._prepare_backbone(changed)
    assert model._backbone is not backbone
    assert model._backbone.get_classifier().weight.shape[0] == len(changed)


def test_a_first_run_builds_the_backbone():
    model = StubClassifier(device="cpu")
    model._prepare_backbone(["cat", "dog"])
    assert model._backbone.get_classifier().weight.shape[0] == 2


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def test_a_checkpoint_carries_its_classes(tmp_path):
    saved = stub(["cat", "dog"])
    path = tmp_path / "round_001.pt"
    saved.save(path)

    loaded = StubClassifier(device="cpu")
    loaded.load(path)
    assert loaded.classes == ["cat", "dog"]


def test_hyperparameters_come_from_the_constructor_not_the_checkpoint(tmp_path):
    saved = StubClassifier(num_epochs=9, batch_size=64, lr=0.5, device="cpu")
    saved._backbone = saved._build_backbone(2)
    saved.classes = ["cat", "dog"]
    path = tmp_path / "round_001.pt"
    saved.save(path)

    # The checkpoint's config is provenance: letting it win made editing
    # [model.params] look like it did nothing
    loaded = StubClassifier(num_epochs=3, batch_size=16, lr=5e-5, device="cpu")
    loaded.load(path)
    assert (loaded.num_epochs, loaded.batch_size, loaded.lr) == (3, 16, 5e-5)


def test_loaded_weights_match_what_was_saved(tmp_path):
    saved = stub(["cat", "dog"])
    weight = saved._backbone.get_classifier().weight.data.clone()
    path = tmp_path / "round_001.pt"
    saved.save(path)

    loaded = StubClassifier(device="cpu")
    loaded.load(path)
    assert torch.equal(loaded._backbone.get_classifier().weight.data, weight)


def test_saving_without_a_model_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="No model to save"):
        StubClassifier(device="cpu").save(tmp_path / "round_001.pt")
